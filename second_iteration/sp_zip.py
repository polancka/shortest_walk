# -*- coding: utf-8 -*-

import zipfile
import os
import sys
import subprocess
import logging
import tempfile
import shutil

# Setup logger
logger = logging.getLogger("sp_wrapper_logger")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(message)s')

file_handler = logging.FileHandler("log.txt", mode='w', encoding='utf-8')
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def run_sp_py(txt_path):
    try:
        result = subprocess.run(
            [sys.executable, "sp.py", txt_path],
            capture_output=True, text=True, check=True
        )
        output = result.stdout
        for line in output.splitlines():
            if line.startswith("Execution time:"):
                return line
        return "Execution time not found."
    except subprocess.CalledProcessError as e:
        if "ZeroDivisionError" in e.stderr:
            return "ZeroDivisionError"
        return f"Subprocess failed: {e.stderr.strip() or str(e)}"


def process_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        folders = {}
        for name in z.namelist():
            if name.endswith('.txt'):
                parts = name.split('/')
                if len(parts) >= 2:
                    folder = parts[0]
                    folders.setdefault(folder, []).append(name)

        sorted_folders = sorted(folders.keys(), key=lambda x: int(x) if x.isdigit() else x)
        total_files = sum(len(folders[folder]) for folder in sorted_folders)
        processed_files = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            for i, folder in enumerate(sorted_folders):
                is_last_folder = (i == len(sorted_folders) - 1)
                folder_prefix = "└───" if is_last_folder else "├───"
                logger.info(f"{folder_prefix}{folder}")

                files = sorted(folders[folder])
                for j, filename in enumerate(files):
                    processed_files += 1
                    percent = (processed_files / total_files) * 100
                    os.system(f"title Processing {processed_files}/{total_files} ({percent:.2f}%)")

                    is_last_file = (j == len(files) - 1)
                    file_prefix = "        " if is_last_folder else "│       "
                    base_name = os.path.basename(filename)

                    extracted_path = os.path.join(temp_dir, base_name)
                    with open(extracted_path, 'wb') as f_out, z.open(filename) as f_in:
                        shutil.copyfileobj(f_in, f_out)

                    log_prefix = f"{file_prefix}{base_name} -> "
                    exec_time_line = run_sp_py(extracted_path)
                    logger.info(log_prefix + exec_time_line)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_zip(sys.argv[1])
    else:
        logger.info("Usage: python sp_zip.py archive.zip")