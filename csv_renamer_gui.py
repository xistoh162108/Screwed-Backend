#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV Renamer GUI
----------------
- Select multiple CSV files
- Extract parameter title (e.g., "Dew-Frost-Point-at-2-Meters") from header/metadata
- Extract YEAR from CSV content (YEAR column; if multiple, show range "YYYY-YYYY")
- Enter a country code (e.g., EGY)
- Preview and rename to: <COUNTRY>_<YEAR or RANGE>_<Parameter>.csv

Tested with typical NASA POWER regional daily CSVs.
"""

import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import pandas as pd
except Exception:
    pd = None

INVALID_FILENAME_CHARS = r'[\\/:*?"<>|]'
INVALID_REGEX = re.compile(INVALID_FILENAME_CHARS)


def sanitize_filename(name: str) -> str:
    """Make a safe filename by replacing forbidden characters and trimming spaces."""
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.replace('/', '-').replace('\\', '-')
    name = INVALID_REGEX.sub('-', name)
    return name


def extract_parameter_from_text(text: str) -> str | None:
    """
    Updated parser:
      - Always drops the FIRST TWO tokens (e.g., code + source like "T2MDEW  MERRA-2 ..."
        or "SYN1deg  MERRA-2 ...").
      - Uses the remaining tokens as the parameter description.
      - Replaces spaces with hyphens in the final parameter for filename safety.
    """
    # Remove trailing units in parentheses e.g., (C), (W/m^2), etc.
    text = re.sub(r'\s*\([^)]*\)\s*$', '', text).strip()

    parts = text.split()
    # Strictly drop the first TWO tokens
    if len(parts) <= 2:
        return None
    parts = parts[2:]

    desc = " ".join(parts).strip()
    if not desc:
        return None

    # Replace spaces with hyphens for safe filenames
    desc = desc.replace(" ", "-")
    return desc


def read_head_lines(path: str, max_lines: int = 50) -> list[str]:
    lines = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip('\n'))
    except UnicodeDecodeError:
        with open(path, 'r', encoding='latin-1', errors='replace') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip('\n'))
    return lines


def extract_parameter(path: str) -> str | None:
    lines = read_head_lines(path, max_lines=60)

    # Pattern: "<CODE>     <SOURCE> <Description> (Units)"
    for ln in lines:
        if re.search(r'^[A-Z0-9_]{2,}\s{2,}.+\([^)]*\)', ln):
            p = extract_parameter_from_text(ln)
            if p:
                return p

    # Fallback: lines that include the word "PARAMETER"
    for ln in lines:
        if 'PARAMETER' in ln.upper():
            p = extract_parameter_from_text(ln)
            if p:
                return p

    # CSV header line
    for ln in lines:
        if ',' in ln and (not ln.startswith('#')):
            cols = [c.strip() for c in ln.split(',')]
            # Try to find a descriptive column from the end
            for c in cols[::-1]:
                if any(tok in c for tok in ['MERRA', 'Dew', 'Temperature', 'Irradiance', 'Point', 'Humidity']):
                    p = extract_parameter_from_text(c) or c
                    return p
            if len(cols) > 0:
                fallback = cols[-1]
                return extract_parameter_from_text(fallback) or fallback

    return None


def extract_year(path: str) -> str | None:
    if pd is not None:
        try:
            try:
                df = pd.read_csv(path, engine='python')
            except Exception:
                df = pd.read_csv(path, engine='python', comment='#')
            year_col = None
            for col in df.columns:
                if col.strip().upper() == 'YEAR':
                    year_col = col
                    break
            if year_col is not None:
                years = sorted(set(int(y) for y in df[year_col].dropna().astype(int).tolist()))
                if len(years) == 1:
                    return str(years[0])
                elif len(years) > 1:
                    return f"{years[0]}-{years[-1]}"
        except Exception:
            pass

    # Fallback: infer from filename
    fname = os.path.basename(path)
    years = re.findall(r'(?:19|20)\d{2}', fname)
    if years:
        years = sorted(set(int(y) for y in years))
        if len(years) == 1:
            return str(years[0])
        elif len(years) > 1:
            return f"{years[0]}-{years[-1]}"
    return None


def propose_new_name(country: str, year: str | None, param: str | None) -> str:
    country = sanitize_filename(country.upper())
    param = sanitize_filename(param) if param else 'UnknownParameter'
    year = year or 'UnknownYear'
    return f"{country}_{year}_{param}.csv"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSV Renamer")
        self.geometry("850x560")
        self.minsize(760, 480)

        self.selected_files: list[str] = []

        top = ttk.Frame(self, padding=10)
        top.pack(fill='x')

        ttk.Label(top, text="Country Code (e.g., EGY):").grid(row=0, column=0, sticky='w')
        self.country_var = tk.StringVar()
        self.country_entry = ttk.Entry(top, textvariable=self.country_var, width=12)
        self.country_entry.grid(row=0, column=1, padx=(6, 20))

        self.btn_add = ttk.Button(top, text="Add CSV Files…", command=self.add_files)
        self.btn_add.grid(row=0, column=2, padx=5)

        self.btn_clear = ttk.Button(top, text="Clear List", command=self.clear_files)
        self.btn_clear.grid(row=0, column=3, padx=5)

        self.btn_preview = ttk.Button(top, text="Preview", command=self.refresh_preview)
        self.btn_preview.grid(row=0, column=4, padx=5)

        self.btn_rename = ttk.Button(top, text="Rename Files", command=self.rename_files)
        self.btn_rename.grid(row=0, column=5, padx=5)

        cols = ("original", "parameter", "year", "newname")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=16)
        self.tree.heading("original", text="Original File")
        self.tree.heading("parameter", text="Parameter")
        self.tree.heading("year", text="YEAR")
        self.tree.heading("newname", text="Proposed New Name")
        self.tree.column("original", width=320, anchor='w')
        self.tree.column("parameter", width=220, anchor='w')
        self.tree.column("year", width=80, anchor='center')
        self.tree.column("newname", width=320, anchor='w')
        self.tree.pack(fill='both', expand=True, padx=10, pady=(0,10))

        self.status = tk.StringVar(value="Add CSV files to begin.")
        statusbar = ttk.Label(self, textvariable=self.status, anchor='w', relief='sunken')
        statusbar.pack(fill='x', side='bottom')

        style = ttk.Style(self)
        if sys.platform == "darwin":
            default_font = ("Helvetica", 12)
            self.option_add("*Font", default_font)

    def add_files(self):
        fpaths = filedialog.askopenfilenames(
            title="Select CSV files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        added = 0
        for p in fpaths:
            if p not in self.selected_files:
                self.selected_files.append(p)
                added += 1
        self.status.set(f"Added {added} files. Total: {len(self.selected_files)}")
        self.refresh_preview()

    def clear_files(self):
        self.selected_files.clear()
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.status.set("List cleared.")

    def refresh_preview(self):
        for i in self.tree.get_children():
            self.tree.delete(i)

        country = self.country_var.get().strip() or "XXX"

        for path in self.selected_files:
            try:
                param = extract_parameter(path)
            except Exception:
                param = None
            try:
                year = extract_year(path)
            except Exception:
                year = None

            newname = propose_new_name(country, year, param)
            self.tree.insert("", "end", values=(os.path.basename(path), param or "—", year or "—", newname))

        self.status.set("Preview updated.")

    def rename_files(self):
        country = self.country_var.get().strip()
        if not country:
            messagebox.showwarning("Missing Country Code", "Please enter a country code (e.g., EGY).")
            self.country_entry.focus_set()
            return

        initialdir = os.path.dirname(self.selected_files[0]) if self.selected_files else os.getcwd()
        outdir = filedialog.askdirectory(title="Choose output folder", initialdir=initialdir)
        if not outdir:
            return

        renamed, errors = 0, []
        for item_id, path in zip(self.tree.get_children(), self.selected_files):
            vals = self.tree.item(item_id, "values")
            _, parameter, year, newname = vals
            if parameter in ("—", "", None):
                parameter = None
            if year in ("—", "", None):
                year = None
            target_name = propose_new_name(country, year, parameter)
            src = path
            dst = os.path.join(outdir, target_name)
            try:
                base, ext = os.path.splitext(dst)
                ctr = 1
                final_dst = dst
                while os.path.exists(final_dst):
                    ctr += 1
                    final_dst = f"{base} ({ctr}){ext}"
                with open(src, 'rb') as fsrc, open(final_dst, 'wb') as fdst:
                    fdst.write(fsrc.read())
                renamed += 1
            except Exception as e:
                errors.append((os.path.basename(path), str(e)))

        self.status.set(f"Renamed/copied {renamed} file(s) to: {outdir}")
        if errors:
            msg = "Some files could not be renamed:\n\n" + "\n".join(f"- {name}: {err}" for name, err in errors[:10])
            messagebox.showerror("Errors occurred", msg)


if __name__ == "__main__":
    app = App()
    app.mainloop()
