# PyQt6 GUI Application

A modern desktop application built with PyQt6.

## Project Structure

```
gui/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── venv/                  # Virtual environment
├── ui/                    # UI components
│   ├── __init__.py
│   └── main_window.py     # Main application window
├── utils/                 # Utility functions
│   ├── __init__.py
│   └── helpers.py         # Helper functions
└── resources/             # Resources (styles, icons, etc.)
    └── styles.qss         # Qt Style Sheets
```

## Setup

### 1. Activate Virtual Environment

**Windows:**
```bash
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python main.py
```

## Features

- Modern PyQt6 interface
- Menu bar with File, Edit, and Help menus
- Toolbar with quick actions
- Status bar for notifications
- Custom styling with QSS (Qt Style Sheets)
- Modular structure for easy expansion

## Development

### Adding New Windows

1. Create a new file in `ui/` directory
2. Inherit from `QWidget` or `QDialog`
3. Import and use in your application

### Adding Custom Styles

Edit `resources/styles.qss` to customize the appearance of your application.

### Adding Utility Functions

Add new helper functions in `utils/helpers.py` or create new utility modules.

## Building Executable (Optional)

To create a standalone executable:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed main.py
```

## License

Your license here

## Author

Your name here
