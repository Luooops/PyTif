# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os.path

# Settings for dmgbuild
filename = "dist/PyTif.dmg"
volume_name = "PyTif"
format = "UDZO"

# The application to include
# Important: This assumes PyInstaller has already run and created dist/PyTif.app
app_name = "PyTif.app"
app_path = os.path.join("dist", app_name)

# Icons and layout
files = [app_path]
symlinks = {"Applications": "/Applications"}

# You can customize the icon size, background, etc.
icon_size = 128
icon_locations = {app_name: (140, 120), "Applications": (460, 120)}
window_rect = ((100, 100), (600, 400))
