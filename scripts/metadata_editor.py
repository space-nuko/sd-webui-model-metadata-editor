import os
import glob
import zipfile
import json
import stat
import sys
import io
import inspect
import base64
import shutil
import re
import platform
import subprocess as sp
from collections import OrderedDict
from multiprocessing.pool import ThreadPool as Pool
import tqdm
from PIL import PngImagePlugin, Image

import torch

import modules.scripts as scripts
from modules import shared, script_callbacks
import gradio as gr

from modules.processing import Processed, process_images
from modules import sd_models, hashes
import modules.ui
from modules.ui_components import ToolButton
import modules.extras
import modules.generation_parameters_copypaste as parameters_copypaste

from scripts import lora_compvis, safetensors_hack


folder_symbol = '\U0001f4c2'  # 📂


LORA_MODEL_EXTS = [".pt", ".ckpt", ".safetensors"]
lora_models = {}             # "My_Lora(abcd1234)" -> "C:/path/to/model.safetensors"


def traverse_all_files(curr_path, model_list):
  f_list = [(os.path.join(curr_path, entry.name), entry.stat()) for entry in os.scandir(curr_path)]
  for f_info in f_list:
    fname, fstat = f_info
    if os.path.splitext(fname)[1] in LORA_MODEL_EXTS:
      model_list.append(f_info)
    elif stat.S_ISDIR(fstat.st_mode):
      model_list = traverse_all_files(fname, model_list)
  return model_list


def get_model_hash(metadata, filename):
  if not metadata:
    return hashes.calculate_sha256(filename)

  if "sshs_model_hash" in metadata:
    return metadata["sshs_model_hash"]

  return safetensors_hack.hash_file(filename)


def get_legacy_hash(metadata, filename):
  if not metadata:
    return sd_models.model_hash(filename)

  if "sshs_legacy_hash" in metadata:
    return metadata["sshs_legacy_hash"]

  return sd_models.model_hash(filename)


import filelock
cache_filename = os.path.join(scripts.basedir(), "hashes.json")
cache_data = None

def cache(subsection):
    global cache_data

    if cache_data is None:
        with filelock.FileLock(cache_filename+".lock"):
            if not os.path.isfile(cache_filename):
                cache_data = {"version": 1}
            else:
                with open(cache_filename, "r", encoding="utf8") as file:
                    cache_data = json.load(file)

    s = cache_data.get(subsection, {})
    cache_data[subsection] = s

    return s


def dump_cache():
    with filelock.FileLock(cache_filename+".lock"):
        with open(cache_filename, "w", encoding="utf8") as file:
            json.dump(cache_data, file, indent=4)



def is_safetensors(filename):
    return os.path.splitext(filename)[1] == ".safetensors"


def get_model_rating(filename):
  if not is_safetensors(filename):
    return 0

  metadata = safetensors_hack.read_metadata(filename)
  return int(metadata.get("ssmd_rating", "0"))


def hash_model_file(finfo):
  filename = finfo[0]
  stat = finfo[1]
  name = os.path.splitext(os.path.basename(filename))[0]

  # Prevent a hypothetical "None.pt" from being listed.
  if name != "None":
    metadata = None

    cached = cache("hashes").get(filename, None)
    if cached is None or stat.st_mtime != cached["mtime"]:
      if metadata is None and is_safetensors(filename):
        metadata = safetensors_hack.read_metadata(filename)
      #model_hash = get_model_hash(metadata, filename)
      legacy_hash = get_legacy_hash(metadata, filename)
    else:
      #model_hash = cached["model"]
      legacy_hash = cached["legacy"]

  return {"model": None, "legacy": legacy_hash, "fileinfo": finfo}


def get_all_models(paths, sort_by, filter_by):
  fileinfos = []
  for path in paths:
    if os.path.isdir(path):
      fileinfos += traverse_all_files(path, [])

  print("[MetadataEditor] Updating model hashes...")
  data = []
  thread_count = max(1, int(shared.opts.data.get("metadata_editor_hash_thread_count", 1)))
  p = Pool(processes=thread_count)
  with tqdm.tqdm(total=len(fileinfos)) as pbar:
      for res in p.imap_unordered(hash_model_file, fileinfos):
          pbar.update()
          data.append(res)
  p.close()

  cache_hashes = cache("hashes")

  res = OrderedDict()
  filter_by = filter_by.strip(" ")
  if len(filter_by) != 0:
    data = [x for x in data if filter_by.lower() in os.path.basename(x["fileinfo"][0]).lower()]
  if sort_by == "name":
    data = sorted(data, key=lambda x: os.path.basename(x["fileinfo"][0]))
  elif sort_by == "date":
    data = sorted(data, key=lambda x: -x["fileinfo"][1].st_mtime)
  elif sort_by == "path name":
    data = sorted(data, key=lambda x: x["fileinfo"][0])
  elif sort_by == "rating":
    data = sorted(data, key=lambda x: get_model_rating(x["fileinfo"][0]), reverse=True)

  for result in data:
    finfo = result["fileinfo"]
    filename = finfo[0]
    stat = finfo[1]
    #model_hash = result["model"]
    legacy_hash = result["legacy"]

    name = os.path.splitext(os.path.basename(filename))[0]

    # Prevent a hypothetical "None.pt" from being listed.
    if name != "None":
      full_name = filename + f"({legacy_hash})"
      #full_name = name + f"({model_hash[0:10]})"
      res[full_name] = filename
      #cache_hashes[filename] = {"model": model_hash, "legacy": legacy_hash, "mtime": stat.st_mtime}
      cache_hashes[filename] = {"legacy": legacy_hash, "mtime": stat.st_mtime}

  return res


def update_lora_models():
  global lora_models
  paths = shared.opts.data.get("metadata_editor_lora_paths", "").split(",")
  sort_by = shared.opts.data.get("metadata_editor_sort_models_by", "name")
  filter_by = shared.opts.data.get("metadata_editor_model_name_filter", "")
  res = get_all_models(paths, sort_by, filter_by)

  lora_models = OrderedDict(**{"None": None}, **res)

  dump_cache()


update_lora_models()


def read_lora_metadata(model_path, module):
  if model_path.startswith("\"") and model_path.endswith("\""):             # trim '"' at start/end
    model_path = model_path[1:-1]
  if not os.path.exists(model_path):
    return None

  metadata = None
  if module == "LoRA":
    if os.path.splitext(model_path)[1] == '.safetensors':
      metadata = safetensors_hack.read_metadata(model_path)

  return metadata


def write_lora_metadata(model_path, module, updates):
  if model_path.startswith("\"") and model_path.endswith("\""):             # trim '"' at start/end
    model_path = model_path[1:-1]
  if not os.path.exists(model_path):
    return None

  from safetensors.torch import save_file

  back_up = shared.opts.data.get("metadata_editor_back_up_model_when_saving", True)
  if back_up:
    backup_path = model_path + ".backup"
    if not os.path.exists(backup_path):
      print(f"[MetadataEditor] Backing up current model to {backup_path}")
      shutil.copyfile(model_path, backup_path)

  metadata = None
  tensors = {}
  if module == "LoRA":
    if os.path.splitext(model_path)[1] == '.safetensors':
      tensors, metadata = safetensors_hack.load_file(model_path, "cpu")

      for k, v in updates.items():
        metadata[k] = str(v)

      save_file(tensors, model_path, metadata)
      print(f"[MetadataEditor] Model saved: {model_path}")


def write_webui_model_preview_image(model_path, image):
  basename, ext = os.path.splitext(model_path)
  preview_path = f"{basename}.preview.png"

  # Copy any text-only metadata
  pnginfo_data = PngImagePlugin.PngInfo()
  use_metadata = False
  metadata = PngImagePlugin.PngInfo()
  for key, value in image.info.items():
      if isinstance(key, str) and isinstance(value, str):
          metadata.add_text(key, value)
          use_metadata = True

  image.save(preview_path, "PNG", pnginfo=(metadata if use_metadata else None))


def delete_webui_model_preview_image(model_path):
  basename, ext = os.path.splitext(model_path)
  preview_path = f"{basename}.preview.png"

  if os.path.isfile(preview_path):
    os.unlink(preview_path)


def decode_base64_to_pil(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(encoding)))


def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:

        # Copy any text-only metadata
        use_metadata = False
        metadata = PngImagePlugin.PngInfo()
        for key, value in image.info.items():
            if isinstance(key, str) and isinstance(value, str):
                metadata.add_text(key, value)
                use_metadata = True

        image.save(
            output_bytes, "PNG", pnginfo=(metadata if use_metadata else None)
        )
        bytes_data = output_bytes.getvalue()
    return base64.b64encode(bytes_data)


def on_ui_tabs():
  can_edit = False

  def open_folder(f):
    if not os.path.exists(f):
      print(f'Folder "{f}" does not exist. After you create an image, the folder will be created.')
      return
    elif not os.path.isdir(f):
      print(f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""", file=sys.stderr)
      return

    if not shared.cmd_opts.hide_ui_dir_config:
      path = os.path.normpath(f)
      if platform.system() == "Windows":
        os.startfile(path)
      elif platform.system() == "Darwin":
        sp.Popen(["open", path])
      elif "microsoft-standard-WSL2" in platform.uname().release:
        sp.Popen(["wsl-open", path])
      else:
        sp.Popen(["xdg-open", path])

  with gr.Blocks(analytics_enabled=False) as metadata_editor_interface:
    with gr.Row().style(equal_height=False):
      with gr.Column(variant='panel'):
        with gr.Row():
          module = gr.Dropdown(["LoRA"], label=f"Network module", value="LoRA", interactive=True)
          model = gr.Dropdown(list(lora_models.keys()), label=f"Model", value="None", interactive=True)
          modules.ui.create_refresh_button(model, update_lora_models, lambda: {"choices": list(lora_models.keys())}, "refresh_lora_models")

        with gr.Row():
          model_hash = gr.Textbox("", label="Model hash", interactive=False)
          legacy_hash = gr.Textbox("", label="Legacy hash", interactive=False)
        with gr.Row():
          model_path = gr.Textbox("", label="Model path", interactive=False)
          open_folder_button = ToolButton(value=folder_symbol, elem_id="hidden_element" if shared.cmd_opts.hide_ui_dir_config else "open_folder_metadata_editor")
        with gr.Row():
          with gr.Column():
            gr.HTML(value="Copy metadata to other models in directory")
            copy_metadata_dir = gr.Textbox("", label="Containing directory", placeholder="All models in this directory will receive the selected model's metadata")
            copy_same_session = gr.Checkbox(True, label="Only copy to models with same session ID")
            copy_metadata_button = gr.Button("Copy Metadata", variant="primary")

      with gr.Column():
        with gr.Row():
          display_name = gr.Textbox(value="", label="Name", placeholder="Display name for this model", interactive=can_edit)
          author = gr.Textbox(value="", label="Author", placeholder="Author of this model", interactive=can_edit)
        with gr.Row():
          keywords = gr.Textbox(value="", label="Keywords", placeholder="Activation keywords, comma-separated", interactive=can_edit)
        with gr.Row():
          description = gr.Textbox(value="", label="Description", placeholder="Model description/readme/notes/instructions", lines=15, interactive=can_edit)
        with gr.Row():
          source = gr.Textbox(value="", label="Source", placeholder="Source URL where this model could be found", interactive=can_edit)
        with gr.Row():
          rating = gr.Slider(minimum=0, maximum=10, step=1, label="Rating", value=0, interactive=can_edit)
          tags = gr.Textbox(value="", label="Tags", placeholder="Comma-separated list of tags (\"artist, style, character, 2d, 3d...\")", lines=2, interactive=can_edit)
        with gr.Row():
          editing_enabled = gr.Checkbox(label="Editing Enabled", value=can_edit)
          with gr.Row():
            save_metadata_button = gr.Button("Save Metadata", variant="primary", interactive=can_edit)
        with gr.Row():
          save_output = gr.HTML("")
      with gr.Column():
        with gr.Row():
          cover_image = gr.Image(label="Cover image", elem_id="metadata_editor_cover_image", source="upload", interactive=can_edit, type="pil", image_mode="RGBA").style(height=480)
        with gr.Accordion("Image Parameters", open=False):
          with gr.Row():
            info2 = gr.HTML()
        with gr.Row():
          try:
              send_to_buttons = parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])
          except:
              pass
        with gr.Row():
          metadata_view = gr.JSON(value={}, label="Training parameters")
        with gr.Row(visible=False):
          info1 = gr.HTML()
          img_file_info = gr.Textbox(label="Generate Info", interactive=False, lines=6)

    open_folder_button.click(fn=lambda p: open_folder(os.path.dirname(p)), inputs=[model_path], outputs=[])

    def copy_metadata_to_all(module, model, copy_dir, same_session_only):
      if model == "None":
        return "No model loaded."

      model_path = lora_models.get(model, None)
      if model_path is None:
        return f"Model path not found: {model}"

      model_path = os.path.realpath(model_path)

      if os.path.splitext(model_path)[1] != ".safetensors":
        return "Model is not in .safetensors format."

      if not os.path.isdir(copy_dir):
        return "Please provide a directory containing models in .safetensors format."

      metadata = read_lora_metadata(model_path, module)
      count = 0
      for entry in os.scandir(copy_dir):
        if entry.is_file():
          path = os.path.realpath(os.path.join(copy_dir, entry.name))
          if path != model_path and is_safetensors(path):
            if same_session_only:
              other_metadata = safetensors_hack.read_metadata(path)
              session_id = metadata.get("ss_session_id", None)
              other_session_id = other_metadata.get("ss_session_id", None)
              if session_id is None or other_session_id is None or session_id != other_session_id:
                continue

            updates = {
              "ssmd_cover_images": "[]",
              "ssmd_display_name": "",
              "ssmd_keywords": "",
              "ssmd_author": "",
              "ssmd_source": "",
              "ssmd_description": "",
              "ssmd_rating": "0",
              "ssmd_tags": "",
            }

            for k, v in metadata.items():
              if k.startswith("ssmd_"):
                updates[k] = v

            write_lora_metadata(path, module, updates)
            count += 1

      print(f"[MetadataEditor] Updated {count} models in directory {copy_dir}.")
      return f"Updated {count} models in directory {copy_dir}."

    copy_metadata_button.click(fn=copy_metadata_to_all, inputs=[module, model, copy_metadata_dir, copy_same_session], outputs=[save_output])

    def update_editing(enabled):
      updates = [gr.Textbox.update(interactive=enabled)] * 6
      updates.append(gr.Image.update(interactive=enabled))
      updates.append(gr.Slider.update(interactive=enabled))
      return updates
    editing_enabled.change(fn=update_editing, inputs=[editing_enabled], outputs=[display_name, author, source, keywords, description, tags, cover_image, rating])

    cover_image.change(fn=modules.extras.run_pnginfo, inputs=[cover_image], outputs=[info1, img_file_info, info2])

    try:
        parameters_copypaste.bind_buttons(send_to_buttons, cover_image, img_file_info)
    except:
        pass

    def refresh_metadata(module, model):
      if model == "None":
        return {"info": "No model loaded."}, None, "", "", "", "", "", 0, "", "", "", ""

      model_path = lora_models.get(model, None)
      if model_path is None:
        return {"info": f"Model path not found: {model}"}, None, "", "", "", "", "", 0, "", "", "", ""

      if os.path.splitext(model_path)[1] != ".safetensors":
        return {"info": "Model is not in .safetensors format."}, None, "", "", "", "", "", 0, "", "", "", ""

      metadata = read_lora_metadata(model_path, module)

      if metadata is None:
        training_params = {}
        metadata = {}
      else:
        training_params = {k: v for k, v in metadata.items() if k.startswith("ss_")}

      cover_images = json.loads(metadata.get("ssmd_cover_images", "[]"))
      cover_image = None
      if len(cover_images) > 0:
        print(f"[MetadataEditor] Loading embedded cover image.")
        cover_image = decode_base64_to_pil(cover_images[0])
      else:
        basename, ext = os.path.splitext(model_path)
        preview_path = f"{basename}.preview.png"
        if os.path.isfile(preview_path):
          print(f"[MetadataEditor] Loading webui preview image: {preview_path}")
          cover_image = Image.open(preview_path)

      display_name = metadata.get("ssmd_display_name", "")
      author = metadata.get("ssmd_author", "")
      source = metadata.get("ssmd_source", "")
      keywords = metadata.get("ssmd_keywords", "")
      description = metadata.get("ssmd_description", "")
      rating = int(metadata.get("ssmd_rating", "0"))
      tags = metadata.get("ssmd_tags", "")
      model_hash = metadata.get("sshs_model_hash", "???")
      #model_hash = metadata.get("sshs_model_hash", cache("hashes").get(model_path, {}).get("model", ""))
      legacy_hash = metadata.get("sshs_legacy_hash", cache("hashes").get(model_path, {}).get("legacy", ""))

      return training_params, cover_image, display_name, author, source, keywords, description, rating, tags, model_hash, legacy_hash, model_path

    model.change(refresh_metadata, inputs=[module, model], outputs=[metadata_view, cover_image, display_name, author, source, keywords, description, rating, tags, model_hash, legacy_hash, model_path])
    model.change(lambda: "", inputs=[], outputs=[copy_metadata_dir])

    def save_metadata(module, model, cover_image, display_name, author, source, keywords, description, rating, tags):
      if model == "None":
        return "No model selected.", "", ""

      model_path = lora_models.get(model, None)
      if model_path is None:
        return f"file not found: {model_path}", "", ""

      if os.path.splitext(model_path)[1] != ".safetensors":
        return "Model is not in .safetensors format", "", ""

      metadata = safetensors_hack.read_metadata(model_path)
      model_hash = safetensors_hack.hash_file(model_path)
      legacy_hash = get_legacy_hash(metadata, model_path)

      # TODO: Support multiple images
      # Blocked on gradio not having a gallery upload option
      # https://github.com/gradio-app/gradio/issues/1379
      cover_images = []
      if cover_image is not None:
        cover_images.append(encode_pil_to_base64(cover_image).decode("ascii"))

      # NOTE: User-specified metadata should NOT be prefixed with "ss_". This is
      # to maintain backwards compatibility with the old hashing method. "ss_"
      # should be used for training parameters that will never be manually
      # updated on the model.
      updates = {
        "ssmd_cover_images": json.dumps(cover_images),
        "ssmd_display_name": display_name,
        "ssmd_author": author,
        "ssmd_source": source,
        "ssmd_keywords": keywords,
        "ssmd_description": description,
        "ssmd_rating": rating,
        "ssmd_tags": tags,
        "sshs_model_hash": model_hash,
        "sshs_legacy_hash": legacy_hash
      }

      write_lora_metadata(model_path, module, updates)
      if cover_image is None:
        delete_webui_model_preview_image(model_path)
      else:
        write_webui_model_preview_image(model_path, cover_image)

      return "Model saved.", model_hash, legacy_hash

    save_metadata_button.click(save_metadata, inputs=[module, model, cover_image, display_name, author, source, keywords, description, rating, tags], outputs=[save_output, model_hash, legacy_hash])

  return [(metadata_editor_interface, "Model Metadata Editor", "metadata_editor")]


def on_ui_settings():
    section = ('metadata_editor', "Model Metadata Editor")
    shared.opts.add_option("metadata_editor_lora_paths", shared.OptionInfo("", "Paths to scan for LoRA models, comma-separated", section=section))
    shared.opts.add_option("metadata_editor_sort_models_by", shared.OptionInfo("name", "Sort models by", gr.Radio, {"choices": ["name", "date", "path name", "rating"]}, section=section))
    shared.opts.add_option("metadata_editor_model_name_filter", shared.OptionInfo("", "Model name filter", section=section))
    shared.opts.add_option("metadata_editor_back_up_model_when_saving", shared.OptionInfo(True, "Make a backup copy of the model being edited when saving its metadata.", section=section))
    shared.opts.add_option("metadata_editor_hash_thread_count", shared.OptionInfo(1, "# of threads to use for hash calculation (increase if using an SSD)", section=section))


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
