# HindBG - AI Background Removal Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

HindBG is an AI-powered background removal tool that leverages advanced computer vision techniques to remove backgrounds from images while preserving fine details—especially delicate hair structures. This project is built with Flask and provides a modern, responsive web interface for processing images.

---

## Table of Contents

- [Overview](#overview)
- [EXE](#exe)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Showcases](#showcases)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)

---

## Overview

HindBG delivers near-perfect background removal using a sophisticated pipeline that includes both a standard background remover and an optional enhanced processor. Users can choose whether to enable the enhanced processing for sharper edges and more detailed output. The tool manages user tokens, image histories, and automatically cleans up temporary files to ensure smooth operation.

---

## EXE

<p>You can download the exe file from the <code>dist</code> folder:</p>
(File is currently not uploaded, but you can use this via intructions mentioned below.)
<a href="dist/app.exe" download style="
    background-color: #007BFF;
    color: white;
    padding: 5px 10px;
    border-radius: 12px;
    text-decoration: none;
    font-weight: bold;
    display: inline-block;
">
    Download EXE
</a>

---

## Features

- **AI-Powered Background Removal:**  
  Uses the transparent-background package powered by PyTorch (64-bit only) to remove backgrounds accurately.

- **Optional Enhanced Processing:**  
  Enhance images before background removal for even crisper output with finer details.

- **User Management & Token System:**  
  Built with Flask, Flask-Login, and Flask-SQLAlchemy for secure user authentication and token-based usage.

- **Responsive Web Interface:**  
  A modern UI built using Tailwind CSS allows drag-and-drop uploads, live processing status, and easy downloads.

- **Automatic File Cleanup:**  
  Old files in upload and processed directories are automatically removed after a specified period.

---

## Requirements

- **Python:** 3.6 or higher
- **Flask:** 2.3.3
- **Flask-SQLAlchemy:** 3.1.1
- **Flask-Login:** 0.6.2
- **Flask-WTF:** 1.1.1
- **Werkzeug:** 2.3.7
- **Pillow:** >=10.1.0
- **transparent-background:** 1.3.3
- **PyTorch:** >=2.2.0 (64-bit only)
- **NumPy:** >=1.24.3
- **tqdm:** >=4.66.1

> **Note:** This project depends on PyTorch, which is only supported on 64-bit systems for Windows.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aseemgupta256/HindBG.git
   cd HindBG
   ```

2. **Create a 64-bit virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Configure Environment Variables:**  
   Customize settings in a `.env` file (e.g., `USE_GPU`, `USE_JIT`).

---

## Usage

1. **Run the Flask application:**
   ```bash
   python app.py
   ```

2. **Access the tool:**  
   Open your web browser and navigate to [http://localhost:8000](http://localhost:8000).

3. **Workflow:**
   - **Register or Login:** Securely access your dashboard.
   - **Upload Images:** Use the drag-and-drop or file select interface on the dashboard.
   - **Select Processing Mode:** Enable or disable enhanced processing using the checkbox.
   - **Process & Download:** Watch the live processing status; after processing, view and download the resulting image.

4. **Password:**
   - **Admin Login:** user:admin, pass:admin123.
   - **User Login:** you can create and add tokens via admin account.
---

## Showcases

### Input Examples
_Showcase some input images here (place your images in a `screenshots` folder or similar):_

<img src="screenshots/1.jpg" alt="Input Example 1" width="200" style="height: auto;" />
<img src="screenshots/2.jpg" alt="Input Example 2" width="200" style="height: auto;" />
<img src="screenshots/3.jpg" alt="Input Example 3" width="200" style="height: auto;" />

### Output Examples
_Showcase processed results here:_

<img src="screenshots/output1.png" alt="Output Example 1" width="200" style="height: auto;" />
<img src="screenshots/output2.png" alt="Output Example 2" width="200" style="height: auto;" />
<img src="screenshots/output3.png" alt="Output Example 3" width="200" style="height: auto;" />


## Contributing

Contributions are welcome! If you have ideas for improvements, feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

## Author

**Aseem Gupta**  
[GitHub Profile](https://github.com/Aseemgupta256)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
