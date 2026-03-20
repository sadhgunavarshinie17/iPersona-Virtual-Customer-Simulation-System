import sys
import pandas as pd
import io
import re
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QTextEdit, QLabel, QLineEdit, QDialog, QMessageBox, 
    QFileDialog, QScrollArea, QInputDialog, QFrame, QSplitter, QGridLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QSizePolicy,
    QTabWidget, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect, QSize
from PyQt6.QtGui import QPixmap, QScreen, QFont, QPalette, QColor, QIcon, QResizeEvent
import os
import markdown

# Import your existing modules
from pipeline import (
    run_clustering_pipeline,
    run_product_loading_pipeline,
    run_review_generation_pipeline,
    run_analysis_pipeline,
    run_analysis_only,
    run_review_generation_only
)
from visualization import plot_cluster_ratings, print_reviews_to_console
from report_generator import save_analysis_report, display_analysis_results, export_to_excel
from advanced_analysis import (
    cluster_comparison_analysis,
    product_performance_analysis,
    market_insights_analysis,
    custom_analysis
)

import os
from pathlib import Path
from visualization import plot_cluster_ratings, print_reviews_to_console
from report_generator import save_analysis_report, display_analysis_results
from pipeline import analyze_reviews_with_gpt4

from report_generator import save_analysis_report, display_analysis_results, export_to_excel
from datetime import datetime

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import openai
import re

from pathlib import Path
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QTableWidgetItem
from PyQt6.QtCore import QThread, pyqtSignal
import traceback

from config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE,
    RATING_METRICS, DEFAULT_RATING, DEFAULT_NUM_REVIEWS_PER_CLUSTER, MAX_THREADS
)

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, 
                             QLineEdit, QPushButton, QLabel, QWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from openai import OpenAI

# Global Data Storage
global_df = None
global_cluster_characteristics = None
global_product_data = None
global_cluster_reviews = None
global_analysis_result = None

def filter_progress_bar_output(text):
    """Filters out progress bar lines from tqdm."""
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        if (re.search(r'\d+/\d+\s*\[.*?(\d+\.\d+%)?.*?\]', line) or
            re.search(r'it/s|s/it', line) or
            '|██' in line or '─' in line or '▏' in line or '▎' in line or
            (line.endswith('\r') and not line.endswith('\n'))
        ):
            continue
        filtered_lines.append(line)
            
    return '\n'.join([line for line in filtered_lines if line.strip()])

def load_data_file(file_path):
    """Load data from CSV or Excel file."""
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            return pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
    except Exception as e:
        raise Exception(f"Error loading file: {str(e)}")

class Worker(QThread):
    """Worker thread for synchronous operations."""
    finished = pyqtSignal(object, str)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        original_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result, captured_output.getvalue())
        except Exception as e:
            self.error.emit(str(e))
        finally:
            sys.stdout = original_stdout
            captured_output.close()

class ModernButton(QPushButton):
    """Custom modern button with hover effects."""
    def __init__(self, text, primary=False):
        super().__init__(text)
        self.primary = primary
        self.setStyleSheet(self.get_style())
        self.setMinimumHeight(45)
        self.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))

    def get_style(self):
        if self.primary:
            return """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                stop:0 #667eea, stop:1 #764ba2);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 12px 24px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                stop:0 #5a6fd8, stop:1 #6a4190);
                    transform: translateY(-2px);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                stop:0 #4c5bc6, stop:1 #5e377e);
                }
                QPushButton:disabled {
                    background: #cccccc;
                    color: #666666;
                }
            """
        else:
            return """
                QPushButton {
                    background: rgba(255, 255, 255, 0.1);
                    color: #333333;
                    border: 2px solid rgba(102, 126, 234, 0.3);
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: rgba(102, 126, 234, 0.1);
                    border-color: rgba(102, 126, 234, 0.6);
                    color: #667eea;
                }
                QPushButton:pressed {
                    background: rgba(102, 126, 234, 0.2);
                }
            """

class ModernTableWidget(QTableWidget):
    """Custom table widget with modern styling."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QTableWidget {
                background-color: rgba(255, 255, 255, 0.98);
                border: none;
                border-radius: 12px;
                gridline-color: rgba(102, 126, 234, 0.2);
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
                selection-background-color: rgba(102, 126, 234, 0.3);
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid rgba(102, 126, 234, 0.1);
            }
            QTableWidget::item:selected {
                background-color: rgba(102, 126, 234, 0.2);
                color: #2d3748;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                            stop:0 #667eea, stop:1 #764ba2);
                color: white;
                padding: 12px 8px;
                border: none;
                font-weight: 600;
                font-size: 11px;
            }
            QHeaderView::section:horizontal {
                border-right: 1px solid rgba(255, 255, 255, 0.2);
            }
            QHeaderView::section:vertical {
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            }
            QScrollBar:vertical {
                background: rgba(0, 0, 0, 0.1);
                width: 12px;
                border-radius: 6px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #667eea;
                border-radius: 6px;
                min-height: 20px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background: #5a67d8;
            }
            QScrollBar:horizontal {
                background: rgba(0, 0, 0, 0.1);
                height: 12px;
                border-radius: 6px;
                margin: 0;
            }
            QScrollBar::handle:horizontal {
                background: #667eea;
                border-radius: 6px;
                min-width: 20px;
                margin: 2px;
            }
        """)
        
        # Set table properties
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSortingEnabled(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)

    def populate_from_dataframe(self, df):
        """Populate table from pandas DataFrame."""
        if df is None or df.empty:
            return
            
        self.setRowCount(len(df))
        self.setColumnCount(len(df.columns))
        self.setHorizontalHeaderLabels(df.columns.tolist())
        
        for row in range(len(df)):
            for col in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iloc[row, col]))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.setItem(row, col, item)
        
        # Auto-resize columns to content
        self.resizeColumnsToContents()
        
        # Set minimum column widths
        for col in range(self.columnCount()):
            if self.columnWidth(col) < 100:
                self.setColumnWidth(col, 100)

class PlotViewer(QDialog):
    """Modern plot viewer with dropdown selection."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📊 Plot Viewer")
        self.setMinimumSize(900, 700)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                            stop:0 #f8f9fa, stop:1 #e9ecef);
            }
            QComboBox {
                padding: 8px 12px;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                background: white;
                font-size: 12px;
                font-weight: 500;
                min-width: 200px;
            }
            QComboBox:hover {
                border-color: #667eea;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border: 2px solid #667eea;
                width: 6px;
                height: 6px;
                border-top: none;
                border-right: none;
                transform: rotate(-45deg);
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                border: 2px solid #667eea;
                border-radius: 8px;
                background: white;
                selection-background-color: rgba(102, 126, 234, 0.2);
                padding: 4px;
            }
            QLabel {
                color: #2d3748;
                font-weight: 600;
            }
        """)
        
        self.setup_ui()
        self.load_available_plots()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header
        header = QLabel("📊 Visualization Gallery")
        header.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("""
            QLabel {
                color: #2d3748;
                padding: 15px;
                background: rgba(102, 126, 234, 0.1);
                border-radius: 12px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(header)

        # Controls
        controls_layout = QHBoxLayout()
        
        plot_label = QLabel("Select Plot:")
        plot_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        controls_layout.addWidget(plot_label)
        
        self.plot_combo = QComboBox()
        self.plot_combo.currentTextChanged.connect(self.load_selected_plot)
        controls_layout.addWidget(self.plot_combo)
        
        controls_layout.addStretch()
        
        refresh_btn = ModernButton("🔄 Refresh", primary=False)
        refresh_btn.clicked.connect(self.load_available_plots)
        controls_layout.addWidget(refresh_btn)
        
        layout.addLayout(controls_layout)

        # Image display
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setStyleSheet("""
            QScrollArea {
                border: 2px solid rgba(102, 126, 234, 0.2);
                border-radius: 12px;
                background: white;
            }
        """)
        
        self.image_label = QLabel("No plot selected")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background: white;
                padding: 40px;
                color: #6c757d;
                font-size: 14px;
                font-style: italic;
            }
        """)
        self.image_label.setScaledContents(False)
        self.image_scroll.setWidget(self.image_label)
        
        layout.addWidget(self.image_scroll)

        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-size: 11px;
                padding: 8px;
                background: rgba(102, 126, 234, 0.05);
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def load_available_plots(self):
        """Load available plot files from both current and plots directory"""
        self.plot_combo.clear()
        plot_files = []
        plot_extensions = ['.png', '.jpg', '.jpeg', '.svg']
        
        # Check both current directory and plots directory
        search_dirs = ['.', 'plots']
        
        for directory in search_dirs:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    full_path = os.path.join(directory, filename)
                    if any(filename.lower().endswith(ext) for ext in plot_extensions):
                        if any(keyword in filename.lower() for keyword in ['cluster', 'plot', 'chart', 'graph', 'analysis']):
                            plot_files.append((filename, full_path))
        
        if plot_files:
            # Sort by modification time (newest first)
            plot_files.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
            for filename, full_path in plot_files:
                self.plot_combo.addItem(filename, full_path)
            self.status_label.setText(f"Found {len(plot_files)} plot files")
            
            # Select the first item by default
            if self.plot_combo.count() > 0:
                self.plot_combo.setCurrentIndex(0)
                self.load_selected_plot(self.plot_combo.currentText())
        else:
            self.plot_combo.addItem("No plots available")
            self.status_label.setText("No plot files found. Run analysis to generate visualizations.")

    def load_selected_plot(self, filename):
        """Load and display selected plot"""
        if not filename or filename == "No plots available":
            self.image_label.setText("No plot selected")
            self.image_label.setPixmap(QPixmap())
            return
            
        full_path = self.plot_combo.currentData()  # Get the full path from item data
        
        try:
            if os.path.exists(full_path):
                pixmap = QPixmap(full_path)
                if not pixmap.isNull():
                    # Calculate maximum size that fits in scroll area
                    max_width = self.image_scroll.width() - 50
                    max_height = self.image_scroll.height() - 50
                    
                    # Scale while maintaining aspect ratio
                    scaled_pixmap = pixmap.scaled(
                        QSize(max_width, max_height),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.image_label.setPixmap(scaled_pixmap)
                    self.image_label.setText("")
                    self.image_label.resize(scaled_pixmap.size())
                    self.status_label.setText(f"Displaying: {filename}")
                else:
                    self.show_error(f"Could not load image from '{filename}'")
            else:
                self.show_error(f"File not found '{filename}'")
        except Exception as e:
            self.show_error(f"Error displaying image: {str(e)}")

    def show_error(self, message):
        """Helper to show error messages"""
        self.image_label.setText(message)
        self.image_label.setPixmap(QPixmap())
        self.status_label.setText("Error loading image")

    def resizeEvent(self, event):
        """Handle resize events to rescale image."""
        super().resizeEvent(event)
        if hasattr(self, 'plot_combo') and self.plot_combo.currentText():
            # Reload current plot with new size
            self.load_selected_plot(self.plot_combo.currentText())
            

class ChatWindow(QDialog):
    """Enhanced ChatGPT-style chat interface."""
    
    def __init__(self, cluster_reviews, cluster_characteristics, parent=None):
        super().__init__(parent)
        self.cluster_reviews = cluster_reviews
        self.cluster_characteristics = cluster_characteristics
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        self.setWindowTitle("Customer Intelligence Assistant")
        self.setMinimumSize(900, 700)
        self.resize(1200, 800)
        
        # Dark theme styling
        self.setStyleSheet("""
            QDialog { 
                background: #1a1a1a; 
                color: #ffffff;
                font-family: 'Segoe UI', sans-serif;
            }
        """)
        
        self.setup_ui()
        self.add_welcome_message()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Modern header with gradient
        header = QLabel("🧠 Customer Intelligence Assistant")
        header.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setFixedHeight(70)
        header.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6366f1, stop:0.5 #8b5cf6, stop:1 #06b6d4);
                color: white;
                border: none;
                padding: 20px;
                font-weight: 600;
            }
        """)
        layout.addWidget(header)

        # Chat area with modern styling
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setStyleSheet("""
            QTextEdit {
                background: #212121;
                border: none;
                padding: 20px;
                font-size: 14px;
                line-height: 1.6;
                color: #e0e0e0;
            }
            QScrollBar:vertical {
                background: #2a2a2a;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #404040;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.chat_area)

        # Input section
        input_container = QWidget()
        input_container.setFixedHeight(80)
        input_container.setStyleSheet("""
            QWidget {
                background: #2a2a2a;
                border-top: 1px solid #404040;
            }
        """)
        
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(20, 15, 20, 15)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask about customer insights, product performance, or recommendations...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background: #1a1a1a;
                border: 1px solid #404040;
                border-radius: 25px;
                padding: 12px 20px;
                font-size: 14px;
                color: #ffffff;
            }
            QLineEdit:focus {
                border: 1px solid #6366f1;
                background: #262626;
            }
        """)
        self.input_field.returnPressed.connect(self.send_message)
        
        send_btn = QPushButton("→")
        send_btn.setFixedSize(50, 50)
        send_btn.setStyleSheet("""
            QPushButton {
                background: #6366f1;
                border: none;
                border-radius: 25px;
                color: white;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #5b5fff;
            }
            QPushButton:pressed {
                background: #4f46e5;
            }
        """)
        send_btn.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(send_btn)
        layout.addWidget(input_container)
        
        self.setLayout(layout)

    def add_welcome_message(self):
        welcome = """
        <div style='max-width: 80%; margin: 20px 0; padding: 25px; background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 16px; border-left: 4px solid #06b6d4;'>
            <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                <span style='font-size: 24px; margin-right: 12px;'>🤖</span>
                <span style='font-size: 16px; font-weight: 600; color: #06b6d4;'>Assistant</span>
            </div>
            <p style='margin: 0 0 15px 0; font-size: 15px; line-height: 1.5; color: #e2e8f0;'>
                Hello! I'm your Customer Intelligence Assistant. I can help you analyze:
            </p>
            <div style='margin: 15px 0; padding-left: 15px;'>
                <div style='margin: 8px 0; color: #94a3b8;'>• Customer segment insights and behaviors</div>
                <div style='margin: 8px 0; color: #94a3b8;'>• Product performance and ratings analysis</div>
                <div style='margin: 8px 0; color: #94a3b8;'>• Market opportunities and recommendations</div>
                <div style='margin: 8px 0; color: #94a3b8;'>• Feature enhancement suggestions</div>
            </div>
            <p style='margin: 15px 0 0 0; font-size: 13px; color: #64748b; font-style: italic;'>
                Try asking: "What features should we improve for students?" or "Which customer segment has the highest satisfaction?"
            </p>
        </div>
        """
        self.chat_area.append(welcome)

    def add_user_message(self, message):
        user_msg = f"""
        <div style='text-align: right; margin: 20px 0;'>
            <div style='display: inline-block; max-width: 70%; padding: 15px 20px; background: #6366f1; color: white; border-radius: 20px 20px 5px 20px; text-align: left; font-size: 14px; line-height: 1.4;'>
                {message.replace(chr(10), '<br>')}
            </div>
        </div>
        """
        self.chat_area.append(user_msg)
        self.scroll_to_bottom()

    def add_assistant_message(self, message):
        formatted_msg = self.format_analysis_response(message)
        
        assistant_msg = f"""
        <div style='margin: 20px 0;'>
            <div style='max-width: 85%; background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 16px; border-left: 4px solid #06b6d4; overflow: hidden;'>
                <div style='padding: 12px 20px; background: rgba(6, 182, 212, 0.1);'>
                    <span style='font-size: 18px; margin-right: 8px;'>🤖</span>
                    <span style='font-weight: 600; color: #06b6d4;'>Assistant</span>
                </div>
                <div style='padding: 20px; color: #e2e8f0; line-height: 1.6;'>
                    {formatted_msg}
                </div>
            </div>
        </div>
        """
        self.chat_area.append(assistant_msg)
        self.scroll_to_bottom()

    def format_analysis_response(self, text):
        """Convert analysis to clean, professional format with prominent headings"""
        if not isinstance(text, str):
            text = str(text)
        
        # Simple formatting - convert line breaks to HTML
        formatted_text = text.replace('\n', '<br>')
        return f"<p style='color: #e2e8f0; margin: 0;'>{formatted_text}</p>"

    def add_thinking_animation(self):
        thinking = """
        <div style='margin: 20px 0;'>
            <div style='display: inline-block; padding: 15px 20px; background: #2a2a2a; border-radius: 20px; color: #94a3b8;'>
                <span style='font-size: 16px; margin-right: 8px;'>🤖</span>
                <span style='animation: pulse 1.5s infinite;'>Analyzing data...</span>
            </div>
        </div>
        """
        self.chat_area.append(thinking)
        self.scroll_to_bottom()

    def send_message(self):
        message = self.input_field.text().strip()
        if not message:
            return

        self.add_user_message(message)
        self.input_field.clear()
        self.add_thinking_animation()
        
        # Use thread to avoid blocking UI
        self.worker = OpenAIWorker(message, self.cluster_reviews, self.cluster_characteristics, self.client)
        self.worker.response_ready.connect(self.handle_openai_response)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

    def handle_openai_response(self, response):
        self.remove_last_message()
        self.add_assistant_message(response)

    def handle_error(self, error_msg):
        self.remove_last_message()
        self.add_assistant_message(f"Sorry, I encountered an error: {error_msg}")

    def remove_last_message(self):
        cursor = self.chat_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.select(cursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()

    def scroll_to_bottom(self):
        scrollbar = self.chat_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class OpenAIWorker(QThread):
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, query, cluster_reviews, cluster_characteristics, client):
        super().__init__()
        self.query = query
        self.cluster_reviews = cluster_reviews
        self.cluster_characteristics = cluster_characteristics
        self.client = client

    def run(self):
        try:
            # Prepare detailed context similar to terminal version
            context = f"Customer Clusters: {str(self.cluster_characteristics)[:1000]}\n\nReviews: {str(self.cluster_reviews)[:1000]}"
            
            # Create a detailed prompt similar to your terminal version
            prompt = (
                "Based on the customer clustering and review data provided, please answer this question:\n\n"
                f"{self.query}\n\n"
                f"Context:\n{context}...\n\n"
                "Provide a detailed, data-driven response with specific insights, analysis, and conclusions."
            )
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert data analyst and market researcher. Provide comprehensive, detailed analysis based on the provided customer data. Structure your response with clear sections, specific insights, and data-driven conclusions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,  # Increased from 500 to allow detailed responses
                temperature=0.7
            )
            
            self.response_ready.emit(response.choices[0].message.content)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

class ProductFormDialog(QDialog):
    """Modern product form dialog."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Apple Product")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                            stop:0 #f8f9fa, stop:1 #e9ecef);
            }
            QLineEdit {
                padding: 8px 12px;
                border: 2px solid #e9ecef;
                border-radius: 6px;
                background: white;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #667eea;
                outline: none;
            }
            QLabel {
                font-weight: 500;
                color: #495057;
            }
        """)

        self.layout = QVBoxLayout()
        self.form_widgets = {}

        # Header
        header = QLabel("Add New Apple Product")
        header.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #333; margin: 20px 0;")
        self.layout.addWidget(header)

        self.product_fields = [
            "Model", "Release Year", "Price (AED)", "Colours Available ", "Weight ",
            "Display Type ", "Display Size ", "Display Resolution ", "Storage",
            "Water/Dust Rating ", "Chip ", "RAM", " Camera ", "Front Camera / aperture",
            "Battery (mAh, Approx.)", "Charging (Wired/Wireless)", "Wireless Video Playback ",
            "Fast-charge capability ", "Connectivity", "Face-ID", "Apple Pay ", "Touch ID"
        ]

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #f8f9fa;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #667eea;
                border-radius: 6px;
            }
        """)
        
        form_content_widget = QWidget()
        form_content_layout = QGridLayout(form_content_widget)
        form_content_layout.setSpacing(15)

        for i, field in enumerate(self.product_fields):
            label = QLabel(f"{field}:")
            input_field = QLineEdit()
            self.form_widgets[field] = input_field
            
            row = i // 2
            col = (i % 2) * 2
            form_content_layout.addWidget(label, row, col)
            form_content_layout.addWidget(input_field, row, col + 1)
        
        scroll_area.setWidget(form_content_widget)
        self.layout.addWidget(scroll_area)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        cancel_btn = ModernButton("Cancel", primary=False)
        cancel_btn.clicked.connect(self.reject)
        
        add_btn = ModernButton("Add Product", primary=True)
        add_btn.clicked.connect(self.accept)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(add_btn)
        
        self.layout.addLayout(button_layout)
        self.setLayout(self.layout)

    def get_product_data(self):
        """Get entered product data as dictionary."""
        return {field: widget.text().strip() for field, widget in self.form_widgets.items()}

class CustomerAnalysisGUI(QMainWindow):
    """Main GUI application for customer analysis."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("iPersona")
        self.setMinimumSize(1200, 800)
        self.cluster_reviews = None
        self.cluster_characteristics = None
        self.analysis_result = None
        self.analysis_completed = False
        
        # Apply modern styling
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                            stop:0 #f8f9fa, stop:1 #e9ecef);
            }
            QTabWidget::pane {
                border: 2px solid rgba(102, 126, 234, 0.2);
                border-radius: 12px;
                background: white;
                margin-top: 10px;
            }
            QTabBar::tab {
                background: rgba(102, 126, 234, 0.1);
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 500;
                color: #495057;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                            stop:0 #667eea, stop:1 #764ba2);
                color: white;
                font-weight: 600;
            }
            QTabBar::tab:hover:!selected {
                background: rgba(102, 126, 234, 0.2);
            }
            QGroupBox {
                font-weight: 600;
                color: #495057;
                border: 2px solid rgba(102, 126, 234, 0.2);
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                background: white;
            }
        """)
        
        self.setup_ui()
        self.setup_status_bar()

    def setup_ui(self):
        """Setup the main UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        

        # Header
        header = QLabel("iPersona")
        header.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("""
            QLabel {
                color: white;
                padding: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                            stop:0 #667eea, stop:1 #764ba2);
                border-radius: 15px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(header)

        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.create_data_tab()
        self.create_analysis_tab()
        self.create_results_tab()
        self.create_advanced_tab()
        
        layout.addWidget(self.tab_widget)

    def create_data_tab(self):
        """Create data management tab."""
        data_tab = QWidget()
        layout = QVBoxLayout(data_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Data Management Group
        data_group = QGroupBox("📊 Data Management")
        data_layout = QVBoxLayout(data_group)
        data_layout.setSpacing(15)

        # File operations
        file_layout = QHBoxLayout()
        
        load_btn = ModernButton("📁 Load Customer Data", primary=True)
        load_btn.clicked.connect(self.load_customer_data)
        
        load_products_btn = ModernButton("🛍️ Load Product Data", primary=False)
        load_products_btn.clicked.connect(self.load_product_data)
        
        add_product_btn = ModernButton("➕ Add New Product", primary=False)
        add_product_btn.clicked.connect(self.add_new_product)
        
        file_layout.addWidget(load_btn)
        file_layout.addWidget(load_products_btn)
        file_layout.addWidget(add_product_btn)
        file_layout.addStretch()
        
        data_layout.addLayout(file_layout)

        # Data Preview
        preview_label = QLabel("📋 Data Preview:")
        preview_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        data_layout.addWidget(preview_label)

        self.data_table = ModernTableWidget()
        self.data_table.setMaximumHeight(300)
        data_layout.addWidget(self.data_table)

        # Data Info
        self.data_info_label = QLabel("No data loaded")
        self.data_info_label.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-size: 12px;
                padding: 10px;
                background: rgba(102, 126, 234, 0.05);
                border-radius: 6px;
            }
        """)
        data_layout.addWidget(self.data_info_label)

        layout.addWidget(data_group)
        layout.addStretch()

        self.tab_widget.addTab(data_tab, "📊 Data")


        # In create_results_tab() method:
        export_btn = ModernButton("📤 Export Results", primary=False)
        export_btn.clicked.connect(self.export_results)  # This connects to the method we modified

    def create_analysis_tab(self):
        """Create analysis tab."""
        analysis_tab = QWidget()
        layout = QVBoxLayout(analysis_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Analysis Pipeline Group
        pipeline_group = QGroupBox("🔬 Analysis Pipeline")
        pipeline_layout = QVBoxLayout(pipeline_group)
        pipeline_layout.setSpacing(15)

        # Remove the Full Pipeline section completely
        # Individual Components
        components_label = QLabel("Individual Components:")
        components_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        pipeline_layout.addWidget(components_label)

        components_layout = QGridLayout()
        
        clustering_btn = ModernButton("🔍 Customer Clustering", primary=False)
        clustering_btn.clicked.connect(self.run_clustering)
        
        review_gen_btn = ModernButton("📝 Generate Reviews", primary=False)
        review_gen_btn.clicked.connect(self.run_review_generation)
        
        analysis_only_btn = ModernButton("📊 Analysis Only", primary=False)
        analysis_only_btn.clicked.connect(self.run_analysis_only)
        
        components_layout.addWidget(clustering_btn, 0, 0)
        components_layout.addWidget(review_gen_btn, 0, 1)
        components_layout.addWidget(analysis_only_btn, 1, 0)
        
        pipeline_layout.addLayout(components_layout)

        layout.addWidget(pipeline_group)

        # Console Output
        console_group = QGroupBox("🖥️ Console Output")
        console_layout = QVBoxLayout(console_group)
        
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setFont(QFont("Consolas", 9))
        self.console_output.setStyleSheet("""
            QTextEdit {
                background: #2d3748;
                color: #e2e8f0;
                border: 1px solid #4a5568;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
        """)
        console_layout.addWidget(self.console_output)

        # Console controls
        console_controls = QHBoxLayout()
        
        clear_console_btn = ModernButton("🗑️ Clear Console", primary=False)
        clear_console_btn.clicked.connect(self.clear_console)
        
        console_controls.addWidget(clear_console_btn)
        console_controls.addStretch()
        
        console_layout.addLayout(console_controls)

        layout.addWidget(console_group)

        self.tab_widget.addTab(analysis_tab, "🔬 Analysis")

    def create_results_tab(self):
        """Create results tab."""
        results_tab = QWidget()
        layout = QVBoxLayout(results_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Results Group
        results_group = QGroupBox("📈 Analysis Results")
        results_layout = QVBoxLayout(results_group)
        results_layout.setSpacing(15)

        # Action buttons
        actions_layout = QHBoxLayout()
        
        generate_results_btn = ModernButton("💾 Generate Graphs", primary=True)
        generate_results_btn.clicked.connect(self.generate_results)
        
        view_plots_btn = ModernButton("📊 View Plots", primary=False)
        view_plots_btn.clicked.connect(self.view_plots)
        
        export_btn = ModernButton("📤 Export Results", primary=False)
        export_btn.clicked.connect(self.export_results)  # This connects to the export method
        
        generate_report_btn = ModernButton("📋 Generate Report", primary=False)
        generate_report_btn.clicked.connect(self.generate_report)
        
        actions_layout.addWidget(generate_results_btn)
        actions_layout.addWidget(view_plots_btn)
        actions_layout.addWidget(export_btn)
        actions_layout.addWidget(generate_report_btn)
        actions_layout.addStretch()
        
        results_layout.addLayout(actions_layout)

        # Results Table
        results_label = QLabel("📊 Cluster Characteristics:")
        results_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        results_layout.addWidget(results_label)

        self.results_table = ModernTableWidget()
        results_layout.addWidget(self.results_table)

        # Customer Reviews Section
        reviews_label = QLabel("📝 Customer Review Analysis:")
        reviews_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        results_layout.addWidget(reviews_label)

        # Create a scrollable text area for the analysis report
        self.review_analysis_display = QTextEdit()
        self.review_analysis_display.setReadOnly(True)
        self.review_analysis_display.setStyleSheet("""
            QTextEdit {
                background: white;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 15px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
                line-height: 1.6;
            }
        """)
        self.review_analysis_display.setPlaceholderText(
            "Review analysis will appear here after generating report..."
        )
        results_layout.addWidget(self.review_analysis_display)

        layout.addWidget(results_group)
        self.tab_widget.addTab(results_tab, "📈 Results")

    def create_advanced_tab(self):
        """Create advanced analysis tab."""
        advanced_tab = QWidget()
        layout = QVBoxLayout(advanced_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Advanced Analysis Group
        advanced_group = QGroupBox("🎯 Advanced Analysis")
        advanced_layout = QVBoxLayout(advanced_group)
        advanced_layout.setSpacing(15)

        # Analysis buttons
        analysis_layout = QGridLayout()
        
        cluster_comparison_btn = ModernButton("🔍 Cluster Comparison", primary=False)
        cluster_comparison_btn.clicked.connect(self.run_cluster_comparison)
        
        product_performance_btn = ModernButton("📊 Product Performance", primary=False)
        product_performance_btn.clicked.connect(self.run_product_performance)
        
        
        custom_chat_btn = ModernButton("🤖 Custom Analysis Chat", primary=True)
        custom_chat_btn.clicked.connect(self.open_custom_chat)
        
        analysis_layout.addWidget(cluster_comparison_btn, 0, 0)
        analysis_layout.addWidget(product_performance_btn, 0, 1)
        analysis_layout.addWidget(custom_chat_btn, 1, 0)
        
        advanced_layout.addLayout(analysis_layout)

        layout.addWidget(advanced_group)

        # Advanced Results
        advanced_results_group = QGroupBox("🔬 Advanced Results")
        advanced_results_layout = QVBoxLayout(advanced_results_group)
        
        self.advanced_results = QTextEdit()
        self.advanced_results.setReadOnly(True)
        self.advanced_results.setStyleSheet("""
            QTextEdit {
                background: white;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 15px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
                line-height: 1.6;
            }
        """)
        advanced_results_layout.addWidget(self.advanced_results)

        layout.addWidget(advanced_results_group)

        self.tab_widget.addTab(advanced_tab, "🎯 Advanced")

    def setup_status_bar(self):
        """Setup status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background: rgba(102, 126, 234, 0.1);
                color: #495057;
                border-top: 1px solid rgba(102, 126, 234, 0.2);
                padding: 5px;
            }
        """)
        self.status_bar.showMessage("Ready")

    def update_status(self, message):
        """Update status bar message."""
        self.status_bar.showMessage(message)

    def append_to_console(self, message):
        """Append message to console/log widget"""
        try:
            if hasattr(self, 'console_output'):
                self.console_output.append(message)
            elif hasattr(self, 'log_text'):
                self.log_text.append(message)
            print(message)  # Always print to terminal as fallback
        except:
            print(message)  # Fail silently, but still print to terminal
        
    def _format_console_text(self, text):
        """Format console text with HTML styling."""
        # Convert to HTML with proper line breaks and styling
        html_lines = []
        for line in text.split('\n'):
            if line.strip():
                # Style different types of lines
                if line.startswith("✅") or line.startswith("✓"):
                    line = f'<span style="color: #48BB78; font-weight: bold;">{line}</span>'  # Green
                elif line.startswith("❌") or line.startswith("⚠️"):
                    line = f'<span style="color: #F56565; font-weight: bold;">{line}</span>'  # Red
                elif line.startswith("🔍") or line.startswith("📊"):
                    line = f'<span style="color: #4299E1; font-weight: bold;">{line}</span>'  # Blue
                elif line.startswith("🚀"):
                    line = f'<span style="color: #9F7AEA; font-weight: bold;">{line}</span>'  # Purple
                elif ":" in line and len(line.split(":")[0]) < 20:  # Likely a label
                    parts = line.split(":", 1)
                    line = f'<span style="color: #F6AD55; font-weight: bold;">{parts[0]}:</span>{parts[1]}'
                
                html_lines.append(line)
        
        # Join with line breaks and wrap in HTML
        html_content = "<br>".join(html_lines)
        return f"""
        <div style="font-family: 'Consolas', 'Monaco', monospace; font-size: 12px; color: #E2E8F0;">
            {html_content}
        </div>
        """

    def clear_console(self):
        """Clear console output."""
        self.console_output.clear()
        # Add some initial HTML structure if needed
        self.console_output.setHtml("""
        <div style="font-family: 'Consolas', 'Monaco', monospace; font-size: 12px; color: #E2E8F0;">
            Console ready. Output will appear here...
        </div>
        """)
        self.update_status("Console cleared")
        
    def load_customer_data(self):
        """Load customer data from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Customer Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                global global_df
                global_df = pd.read_csv(file_path)
                
                # Update data table
                self.data_table.populate_from_dataframe(global_df.head(100))  # Show first 100 rows
                
                # Update info label
                self.data_info_label.setText(
                    f"Loaded {len(global_df)} rows, {len(global_df.columns)} columns from {os.path.basename(file_path)}"
                )
                
                self.update_status(f"Customer data loaded: {len(global_df)} records")
                self.append_to_console(f"✅ Successfully loaded customer data from {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load customer data:\n{str(e)}")
                self.update_status("Error loading customer data")

    def load_product_data(self):
        """Load product data from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Product Data", 
            "", 
            "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                global global_product_data
                global_product_data = load_data_file(file_path)
                self.current_product_file = file_path  # Store the loaded file path
                
                self.update_status(f"Product data loaded: {len(global_product_data)} products")
                self.append_to_console(f"✅ Successfully loaded product data from {file_path}")
                
                # Show data preview
                if hasattr(self, 'data_table'):
                    self.data_table.populate_from_dataframe(global_product_data.head(100))
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load product data:\n{str(e)}")
                self.update_status("Error loading product data")

    def add_new_product(self):
        """Add new product via form dialog."""
        dialog = ProductFormDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            product_data = dialog.get_product_data()
            
            # Add to global product data
            global global_product_data
            if global_product_data is None:
                global_product_data = pd.DataFrame([product_data])
            else:
                global_product_data = pd.concat([global_product_data, pd.DataFrame([product_data])], ignore_index=True)
            
            # Save to file (assuming you want to save to the same file that was loaded)
            try:
                if hasattr(self, 'current_product_file'):
                    file_path = self.current_product_file
                    if file_path.endswith('.csv'):
                        global_product_data.to_csv(file_path, index=False)
                    elif file_path.endswith(('.xlsx', '.xls')):
                        global_product_data.to_excel(file_path, index=False)
                    
                    self.update_status(f"New product added and saved to {os.path.basename(file_path)}")
                else:
                    # If no file was loaded, prompt to save as new file
                    self.save_product_data_as()
                    
                self.append_to_console(f"✅ Added new product: {product_data.get('Model', 'Unknown')}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save product data:\n{str(e)}")
                self.update_status("Error saving product data")

    def save_product_data_as(self):
        """Save product data to a new file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Product Data As",
            "",
            "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                global global_product_data
                if file_path.endswith('.csv'):
                    global_product_data.to_csv(file_path, index=False)
                else:
                    global_product_data.to_excel(file_path, index=False)
                
                self.current_product_file = file_path
                self.update_status(f"Product data saved to {os.path.basename(file_path)}")
                return True
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save product data:\n{str(e)}")
                self.update_status("Error saving product data")
                return False
        return False

    def run_full_pipeline(self):
        """Run the complete analysis pipeline."""
        if global_df is None:
            QMessageBox.warning(self, "Warning", "Please load customer data first!")
            return
            
        self.update_status("Running full analysis pipeline...")
        self.append_to_console("🚀 Starting full analysis pipeline...")
        
        # Start worker thread
        self.worker = Worker(self.execute_full_pipeline)
        self.worker.finished.connect(self.on_pipeline_finished)
        self.worker.error.connect(self.on_pipeline_error)
        self.worker.start()

    def execute_full_pipeline(self):
        """Execute the full analysis pipeline."""
        global global_df, global_cluster_characteristics, global_cluster_reviews, global_analysis_result

        try:
            # 1. Clustering
            df_clustered, cluster_characteristics = run_clustering_pipeline(global_df)
            global_cluster_characteristics = cluster_characteristics
            
            # 2. Product loading
            run_product_loading_pipeline()  # This should set global_product_data
            
            # 3. Review generation
            if global_product_data is not None:
                global_cluster_reviews = run_review_generation_pipeline(
                    df_clustered, 
                    cluster_characteristics
                )
            else:
                raise ValueError("Product data not loaded")
                
            # 4. Analysis
            global_analysis_result = run_analysis_pipeline(
                global_cluster_reviews, 
                global_cluster_characteristics
            )
            
            return "Full pipeline completed successfully"
        except Exception as e:
            raise Exception(f"Pipeline failed: {str(e)}")

    def run_clustering(self):
        """Run clustering analysis with terminal-like output"""
        if global_df is None:
            QMessageBox.warning(self, "Warning", "Please load customer data first!")
            return

        self.update_status("Running customer clustering...")
        self.append_to_console("🔍 Starting customer clustering analysis...")

        # Create a worker that matches your terminal version exactly
        self.worker = Worker(run_clustering_pipeline)  # No parameters, no wrapper
        self.worker.finished.connect(self.on_clustering_finished_simple)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def on_clustering_finished_simple(self, result, output):
        """Just show the output like terminal, don't process results"""
        self.append_to_console(output)
        self.update_status("Clustering completed")
        
        # Optional: If you want to store the results but not display them
        global global_cluster_characteristics
        if result and len(result) > 1:
            global_cluster_characteristics = result[1]  # Store characteristics silently

    def run_review_generation(self):
        """Run review generation only."""
        if global_cluster_characteristics is None:
            QMessageBox.warning(self, "Warning", "Please run clustering analysis first!")
            return
            
        self.update_status("Generating customer reviews...")
        self.append_to_console("📝 Starting review generation...")
        
        self.worker = Worker(run_review_generation_only, global_cluster_characteristics)
        self.worker.finished.connect(self.on_review_generation_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def run_analysis_only(self):
        """Run analysis only without requiring previous steps"""
        self.update_status("Running analysis...")
        self.append_to_console("📊 Starting analysis...")
        
        # Create a wrapper that matches the pipeline's expected signature
        def analysis_wrapper():
            return run_analysis_only()  # No parameters
            
        self.worker = Worker(analysis_wrapper)
        self.worker.finished.connect(self.on_analysis_only_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def run_cluster_comparison(self):
        """Run cluster comparison analysis."""
        if global_cluster_reviews is None or global_cluster_characteristics is None:
            QMessageBox.warning(self, "Warning", "Please run the analysis pipeline first!")
            return
            
        self.update_status("Running cluster comparison analysis...")
        self.append_to_console("🔍 Starting cluster comparison analysis...")
        
        self.worker = Worker(cluster_comparison_analysis, global_cluster_reviews, global_cluster_characteristics)
        self.worker.finished.connect(self.on_advanced_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def run_product_performance(self):
        """Run product performance analysis."""
        if global_cluster_reviews is None or global_cluster_characteristics is None:
            QMessageBox.warning(self, "Warning", "Please run the analysis pipeline first!")
            return
            
        self.update_status("Running product performance analysis...")
        self.append_to_console("📊 Starting product performance analysis...")
        
        self.worker = Worker(product_performance_analysis, global_cluster_reviews)
        self.worker.finished.connect(self.on_advanced_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def run_market_insights(self):
        """Run market insights analysis."""
        if global_cluster_reviews is None or global_cluster_characteristics is None:
            QMessageBox.warning(self, "Warning", "Please run the analysis pipeline first!")
            return
            
        self.update_status("Running market insights analysis...")
        self.append_to_console("💡 Starting market insights analysis...")
        
        self.worker = Worker(market_insights_analysis, global_cluster_reviews, global_cluster_characteristics)
        self.worker.finished.connect(self.on_advanced_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def open_custom_chat(self):
        """Open custom analysis chat window."""
        if global_cluster_reviews is None or global_cluster_characteristics is None:
            QMessageBox.warning(self, "Warning", "Please run the analysis pipeline first!")
            return
            
        chat_window = ChatWindow(global_cluster_reviews, global_cluster_characteristics, self)
        chat_window.exec()

    def view_plots(self):
        """Open plot viewer with original GUI styling but improved functionality"""
        try:
            # Create non-modal dialog that stays on top (original behavior)
            self.plot_viewer = PlotViewer(self)
            self.plot_viewer.setWindowModality(Qt.WindowModality.NonModal)
            self.plot_viewer.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            
            # Modify the PlotViewer's methods to look in output directory
            def custom_load_available_plots():
                """Modified version that only checks output directory"""
                self.plot_viewer.plot_combo.clear()
                plot_files = []
                plot_extensions = ['.png', '.jpg', '.jpeg', '.svg']
                output_dir = Path("output")
                
                if output_dir.exists():
                    for filename in os.listdir(output_dir):
                        full_path = output_dir / filename
                        if any(filename.lower().endswith(ext) for ext in plot_extensions):
                            plot_files.append((filename, full_path))
                
                if plot_files:
                    # Sort by modification time (newest first)
                    plot_files.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
                    for filename, full_path in plot_files:
                        self.plot_viewer.plot_combo.addItem(filename, str(full_path))
                    self.plot_viewer.status_label.setText(f"Found {len(plot_files)} plot files")
                    
                    # Select the first item by default
                    if self.plot_viewer.plot_combo.count() > 0:
                        self.plot_viewer.plot_combo.setCurrentIndex(0)
                        self.plot_viewer.load_selected_plot(self.plot_viewer.plot_combo.currentText())
                else:
                    self.plot_viewer.plot_combo.addItem("No plots available")
                    self.plot_viewer.status_label.setText("No plot files found in output directory")

            # Replace the original method with our modified version
            self.plot_viewer.load_available_plots = custom_load_available_plots
            
            # Refresh and show (original behavior)
            self.plot_viewer.load_available_plots()
            self.plot_viewer.show()
            self.plot_viewer.raise_()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open plot viewer: {str(e)}")
            
        

    def export_results(self):
        """Export analysis results to Excel"""
        global global_analysis_result, global_cluster_reviews
        
        self.append_to_console("🔍 Checking for analysis results...")
        
        # Check for analysis results
        cluster_reviews = global_cluster_reviews or self.cluster_reviews
        analysis_result = global_analysis_result or self.analysis_result
        
        if not cluster_reviews and not analysis_result:
            QMessageBox.warning(
                self, 
                "No Analysis Results", 
                "No analysis results found to export!\n\n"
                "Please follow these steps:\n"
                "1. Load your data in the Data tab\n"
                "2. Run the clustering analysis\n"
                "3. Generate cluster reviews\n"
                "4. Click 'Generate Results' to create visualizations\n"
                "5. Click 'Generate Report' for GPT-4 analysis\n"
                "6. Then try exporting again"
            )
            self.append_to_console("❌ No analysis results available for export")
            return
        
        # Get save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Analysis Results",
            "customer_analysis_results.xlsx",
            "Excel Files (*.xlsx);;All Files (*)"
        )
        
        if file_path:
            try:
                if not file_path.endswith('.xlsx'):
                    file_path += '.xlsx'
                
                self.append_to_console(f"📊 Starting export to: {file_path}")
                self.update_status("Exporting results to Excel...")
                
                # Import your working export function
                from report_generator import export_to_excel_legacy
                
                # Use your existing working export function
                success = export_to_excel_legacy(cluster_reviews, analysis_result, file_path)
                
                if success:
                    # Create detailed success message
                    sheets_info = []
                    if cluster_reviews:
                        sheets_info.append("• Reviews sheet with cluster data")
                        sheets_info.append("• Ratings sheet with extracted metrics")
                    if analysis_result:
                        sheets_info.append("• GPT4_Analysis sheet with comprehensive insights")
                    
                    success_message = f"Analysis results exported successfully!\n\n"
                    success_message += f"📁 File saved to:\n{file_path}\n\n"
                    success_message += f"📊 The Excel file contains:\n"
                    success_message += "\n".join(sheets_info)
                    
                    QMessageBox.information(self, "Export Successful", success_message)
                    
                    self.append_to_console("✅ Export completed successfully!")
                    self.append_to_console(f"📁 File saved: {file_path}")
                    self.update_status("Export completed successfully")
                    
                    # Option to open the file
                    reply = QMessageBox.question(
                        self, 
                        "Open File?", 
                        "Would you like to open the exported Excel file?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        try:
                            import os
                            import platform
                            
                            system = platform.system()
                            if system == "Windows":
                                os.startfile(file_path)
                            elif system == "Darwin":  # macOS
                                os.system(f"open '{file_path}'")
                            else:  # Linux
                                os.system(f"xdg-open '{file_path}'")
                                
                        except Exception as e:
                            self.append_to_console(f"⚠️ Could not open file automatically: {str(e)}")
                            QMessageBox.information(
                                self, 
                                "File Location", 
                                f"File saved successfully but could not be opened automatically.\n\n"
                                f"Please navigate to:\n{file_path}"
                            )
                else:
                    QMessageBox.critical(
                        self, 
                        "Export Failed", 
                        "Failed to export results.\n\n"
                        "Please check the console for error details."
                    )
                    self.append_to_console("❌ Export failed - check error messages above")
                    self.update_status("Export failed")
                    
            except Exception as e:
                error_msg = f"Export error: {str(e)}"
                self.append_to_console(f"❌ {error_msg}")
                self.append_to_console(f"📋 Traceback: {traceback.format_exc()}")
                self.update_status("Export error")
                QMessageBox.critical(self, "Export Error", error_msg)

    def update_results_table(self):
        """Update the results table with cluster characteristics"""
        if not hasattr(self, 'results_table'):
            return
            
        try:
            cluster_characteristics = global_cluster_characteristics or self.cluster_characteristics
            
            if cluster_characteristics:
                # Clear existing data
                self.results_table.setRowCount(0)
                self.results_table.setColumnCount(0)
                
                # Set up table headers
                first_cluster = list(cluster_characteristics.values())[0]
                headers = ['Cluster'] + list(first_cluster.keys())
                self.results_table.setColumnCount(len(headers))
                self.results_table.setHorizontalHeaderLabels(headers)
                
                # Add data rows
                self.results_table.setRowCount(len(cluster_characteristics))
                for row, (cluster_name, characteristics) in enumerate(cluster_characteristics.items()):
                    self.results_table.setItem(row, 0, QTableWidgetItem(str(cluster_name)))
                    for col, (key, value) in enumerate(characteristics.items(), 1):
                        # Format values nicely
                        if isinstance(value, float):
                            formatted_value = f"{value:.2f}"
                        else:
                            formatted_value = str(value)
                        self.results_table.setItem(row, col, QTableWidgetItem(formatted_value))
                
                # Resize columns to content
                self.results_table.resizeColumnsToContents()
                self.append_to_console("📊 Results table updated successfully")
            
        except Exception as e:
            error_msg = f"Error updating results table: {str(e)}"
            self.append_to_console(f"❌ {error_msg}")
            print(f"Table update error: {traceback.format_exc()}")

    def create_test_data(self):
        """Create test data for testing export functionality"""
        try:
            self.append_to_console("🧪 Creating test data...")
            
            # Create sample data that matches your analysis format
            test_cluster_reviews = {
                'Product A': {
                    'Cluster 1': {
                        'review': 'Excellent product! Quality: 5, Price: 4, Service: 5, Delivery: 4',
                        'cluster_info': {
                            'customer_type': 'Premium Customer',
                            'avg_age': 35,
                            'dominant_gender': 'Female',
                            'dominant_occupation': 'Professional',
                            'dominant_income': 'High Income'
                        }
                    },
                    'Cluster 2': {
                        'review': 'Good value for money. Quality: 4, Price: 5, Service: 3, Delivery: 4',
                        'cluster_info': {
                            'customer_type': 'Budget-Conscious',
                            'avg_age': 28,
                            'dominant_gender': 'Male',
                            'dominant_occupation': 'Student',
                            'dominant_income': 'Low Income'
                        }
                    }
                },
                'Product B': {
                    'Cluster 1': {
                        'review': 'Premium experience. Quality: 5, Price: 3, Service: 5, Delivery: 5',
                        'cluster_info': {
                            'customer_type': 'Luxury Seeker',
                            'avg_age': 42,
                            'dominant_gender': 'Female',
                            'dominant_occupation': 'Executive',
                            'dominant_income': 'High Income'
                        }
                    }
                }
            }
            
            test_cluster_characteristics = {
                'Cluster 1': {
                    'avg_age': 35.5,
                    'dominant_gender': 'Female',
                    'customer_count': 150,
                    'avg_rating': 4.2,
                    'avg_quality': 4.8,
                    'avg_price': 3.7,
                    'avg_service': 4.3,
                    'avg_delivery': 4.1
                },
                'Cluster 2': {
                    'avg_age': 28.3,
                    'dominant_gender': 'Male', 
                    'customer_count': 98,
                    'avg_rating': 3.9,
                    'avg_quality': 3.8,
                    'avg_price': 4.5,
                    'avg_service': 3.2,
                    'avg_delivery': 3.9
                }
            }
            
            test_analysis_result = """
CUSTOMER REVIEW ANALYSIS REPORT
===============================

EXECUTIVE SUMMARY:
This comprehensive analysis examines customer feedback across multiple product lines, revealing distinct customer segments and their unique preferences and behaviors.

KEY FINDINGS:

1. CUSTOMER SEGMENTATION:
   • Premium Customers (Cluster 1): High-income professionals prioritizing quality and service
   • Budget-Conscious Customers (Cluster 2): Price-sensitive segment seeking value

2. PRODUCT PERFORMANCE:
   • Product A shows strong performance across all metrics
   • Product B excels in premium customer satisfaction
   • Quality ratings consistently high across segments

3. DEMOGRAPHIC INSIGHTS:
   • Premium segment: Average age 35.5, predominantly female professionals
   • Budget segment: Average age 28.3, mixed gender, students and young professionals

DETAILED ANALYSIS:

CLUSTER 1 - PREMIUM CUSTOMERS (150 customers):
- Demographics: 35.5 years average age, 65% female
- Spending Pattern: High willingness to pay for quality
- Key Drivers: Service excellence, product quality, brand reputation
- Satisfaction: 4.2/5 average rating
- Recommendations: Focus on premium features, personalized service

CLUSTER 2 - BUDGET-CONSCIOUS CUSTOMERS (98 customers):
- Demographics: 28.3 years average age, 55% male
- Spending Pattern: Price-sensitive, value-oriented decisions
- Key Drivers: Competitive pricing, good value proposition
- Satisfaction: 3.9/5 average rating
- Recommendations: Emphasize value, competitive pricing strategies

STRATEGIC RECOMMENDATIONS:

1. PRODUCT STRATEGY:
   - Develop premium product lines for Cluster 1
   - Create value-oriented options for Cluster 2
   - Maintain quality standards across all offerings

2. MARKETING APPROACH:
   - Targeted campaigns for each customer segment
   - Premium messaging for high-income professionals
   - Value-focused communication for price-sensitive customers

3. SERVICE OPTIMIZATION:
   - Enhance premium service options
   - Streamline cost-effective service delivery
   - Implement segment-specific support channels

4. PRICING STRATEGY:
   - Premium pricing for quality-focused customers
   - Competitive pricing for budget-conscious segment
   - Consider tiered pricing models

CONCLUSION:
The analysis reveals clear customer segmentation opportunities. By tailoring products, services, and marketing strategies to these distinct segments, the company can improve customer satisfaction, increase retention, and drive revenue growth.

This test analysis demonstrates the system's capability to generate comprehensive insights from customer review data.
            """
            
            # Store the test data globally and locally
            global global_cluster_reviews, global_cluster_characteristics, global_analysis_result
            global_cluster_reviews = test_cluster_reviews
            global_cluster_characteristics = test_cluster_characteristics
            global_analysis_result = test_analysis_result
            
            # Store using the standard method
            self.store_analysis_results(test_cluster_reviews, test_cluster_characteristics, test_analysis_result)
            
            # Update displays
            self.update_results_table()
            
            if hasattr(self, 'review_analysis_display'):
                self.review_analysis_display.setText(test_analysis_result)
            elif hasattr(self, 'advanced_results'):
                self.advanced_results.setPlainText(test_analysis_result)
            
            self.append_to_console("✅ Test data created successfully!")
            self.append_to_console(f"📊 Created data for {len(test_cluster_reviews)} products")
            self.append_to_console(f"📈 Created {len(test_cluster_characteristics)} cluster profiles")
            self.append_to_console(f"📋 Generated {len(test_analysis_result)} character analysis report")
            self.append_to_console("🔄 You can now test the export functionality")
            
            self.update_status("Test data created successfully")
            
            QMessageBox.information(
                self,
                "Test Data Created",
                "Test data has been created successfully!\n\n"
                "You can now:\n"
                "• View the results table\n" 
                "• See the analysis report\n"
                "• Test the Excel export functionality\n\n"
                "This allows you to verify the export system works correctly."
            )
            
        except Exception as e:
            error_msg = f"Error creating test data: {str(e)}"
            self.append_to_console(f"❌ {error_msg}")
            self.append_to_console(f"📋 Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Error", error_msg)

    def clear_analysis_data(self):
        """Clear all analysis data and reset the interface"""
        global global_analysis_result, global_cluster_reviews, global_cluster_characteristics
        
        try:
            # Clear global variables
            global_analysis_result = None
            global_cluster_reviews = None
            global_cluster_characteristics = None
            
            # Clear instance variables
            self.cluster_reviews = None
            self.cluster_characteristics = None
            self.analysis_result = None
            self.analysis_completed = False
            
            # Clear UI elements
            if hasattr(self, 'results_table'):
                self.results_table.setRowCount(0)
                self.results_table.setColumnCount(0)
            
            if hasattr(self, 'review_analysis_display'):
                self.review_analysis_display.clear()
            elif hasattr(self, 'advanced_results'):
                self.advanced_results.clear()
            
            self.append_to_console("🗑️ Analysis data cleared successfully")
            self.update_status("Analysis data cleared")
            
            QMessageBox.information(
                self,
                "Data Cleared", 
                "All analysis data has been cleared.\n\n"
                "You can now start a new analysis."
            )
            
        except Exception as e:
            error_msg = f"Error clearing data: {str(e)}"
            self.append_to_console(f"❌ {error_msg}")
            QMessageBox.critical(self, "Error", error_msg)

    # Helper methods that may be referenced in your existing code
    def update_status(self, message):
        """Update status bar or status label"""
        try:
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(message)
            elif hasattr(self, 'status_label'):
                self.status_label.setText(message)
            print(f"Status: {message}")  # Always print to console as fallback
        except:
            pass  # Fail silently if no status mechanism exists

    # Helper method to manually set results (for testing)
    def set_test_results(self):
        """Set test results for debugging"""
        global global_analysis_result
        
        test_result = {
            'analysis_result': 'This is a test analysis result',
            'cluster_reviews': {
                'Product A': {
                    'Cluster 1': {
                        'review': 'Test review content',
                        'cluster_info': {
                            'customer_type': 'Regular',
                            'avg_age': 35,
                            'dominant_gender': 'Female',
                            'dominant_occupation': 'Professional',
                            'dominant_income': 'Middle'
                        }
                    }
                }
            }
        }
        
        global_analysis_result = test_result
        self.append_to_console("✅ Test results set - you can now try exporting")


    def generate_results(self):
        """Generate and save results including plots and reviews."""
        global global_cluster_reviews, global_cluster_characteristics, global_analysis_result
        
        if global_cluster_reviews is None or global_cluster_characteristics is None:
            QMessageBox.warning(self, "Warning", "Please run the analysis pipeline first!")
            return
        
        self.update_status("Generating and saving results...")
        self.append_to_console("💾 Generating and saving results...")
        
        try:
            # Create output directory if it doesn't exist
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            # Save plots using plot_cluster_ratings
            plot_path = output_dir / "cluster_ratings.png"
            self.append_to_console(f"📊 Generating cluster ratings plot at: {plot_path}")
            
            # Call the plot function with save_path parameter
            plot_cluster_ratings(
                cluster_reviews=global_cluster_reviews,
                show_plots=False,  # Don't show interactive plots
                save_path=str(plot_path)  # Save to this location
            )
            self.append_to_console(f"✅ Plot saved: {plot_path}")
            
            # Display analysis results if available
            if global_analysis_result is not None:
                analysis_text = display_analysis_results(global_analysis_result)
                self.advanced_results.setPlainText(analysis_text)
                
                # Save analysis report
                report_path = output_dir / "analysis_report.html"
                save_analysis_report(global_analysis_result, str(report_path))
                self.append_to_console(f"📄 Report saved: {report_path}")
            
            # Store results for export
            self.store_analysis_results(
                global_cluster_reviews, 
                global_cluster_characteristics, 
                global_analysis_result
            )
            
            # Update the results table
            self.update_results_table()
            
            self.append_to_console("✅ Results generated successfully")
            self.update_status("Results generated and saved successfully")
            
            # Show success message
            success_msg = f"Results generated successfully!\n\n"
            success_msg += f"📊 Plots saved to: {plot_path}\n"
            if global_analysis_result:
                success_msg += f"📄 Report saved to: {output_dir / 'analysis_report.html'}\n"
            success_msg += f"\n🔄 Results are now ready for export!"
            
            QMessageBox.information(self, "Success", success_msg)
            
        except Exception as e:
            error_msg = f"Error generating results: {str(e)}"
            self.append_to_console(f"❌ {error_msg}")
            self.append_to_console(f"📋 Traceback: {traceback.format_exc()}")
            self.update_status("Error generating results")
            QMessageBox.critical(self, "Error", f"Failed to generate results:\n{str(e)}")

    def display_review_cards(self, cluster_reviews):
        """Display reviews in a card layout."""
        # Clear any existing reviews
        for i in reversed(range(self.reviews_layout.count())): 
            self.reviews_layout.itemAt(i).widget().setParent(None)
        
        # Create scroll area for reviews if it doesn't exist
        if not hasattr(self, 'reviews_scroll'):
            self.reviews_scroll = QScrollArea()
            self.reviews_scroll.setWidgetResizable(True)
            self.reviews_scroll.setStyleSheet("""
                QScrollArea {
                    border: 2px solid rgba(102, 126, 234, 0.1);
                    border-radius: 12px;
                    background: white;
                }
            """)
            
            self.reviews_container = QWidget()
            self.reviews_layout = QVBoxLayout(self.reviews_container)
            self.reviews_layout.setSpacing(15)
            self.reviews_layout.setContentsMargins(15, 15, 15, 15)
            self.reviews_scroll.setWidget(self.reviews_container)
            
            # Add to results tab (you may need to adjust this based on your layout)
            results_tab = self.tab_widget.widget(2)  # Assuming results tab is index 2
            results_tab.layout().insertWidget(1, self.reviews_scroll)  # Insert after first widget
        
        # Add cards for each product and cluster
        for product_name, reviews in cluster_reviews.items():
            # Product header
            product_header = QLabel(f"🍏 {product_name}")
            product_header.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            product_header.setStyleSheet("color: #2d3748; margin-bottom: 5px;")
            self.reviews_layout.addWidget(product_header)
            
            # Add cards for each cluster review
            for cluster_key, review_data in reviews.items():
                cluster_info = review_data['cluster_info']
                
                # Create card widget
                card = QFrame()
                card.setFrameShape(QFrame.Shape.StyledPanel)
                card.setStyleSheet("""
                    QFrame {
                        background: white;
                        border: 1px solid rgba(102, 126, 234, 0.3);
                        border-radius: 12px;
                        padding: 15px;
                    }
                """)
                
                card_layout = QVBoxLayout(card)
                card_layout.setSpacing(10)
                
                # Cluster header
                cluster_header = QLabel(
                    f"👥 {cluster_key} ({cluster_info['customer_type']})"
                )
                cluster_header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
                cluster_header.setStyleSheet("color: #667eea;")
                card_layout.addWidget(cluster_header)
                
                # Profile info
                profile_info = QLabel(
                    f"👤 Profile: {cluster_info['avg_age']:.0f}yr {cluster_info['dominant_gender']}, "
                    f"{cluster_info['dominant_occupation']}, {cluster_info['dominant_income']} income"
                )
                profile_info.setStyleSheet("color: #4a5568; font-size: 11px;")
                card_layout.addWidget(profile_info)
                
                # Review text - update to handle tables
                review_text = review_data['review']
                formatted_review = self._format_analysis_text(review_text)
                
                review_display = QTextEdit()
                review_display.setReadOnly(True)
                review_display.setHtml(f"""
                <html>
                <body style="font-family: 'Segoe UI', Arial, sans-serif; color: #2d3748;">
                    {formatted_review}
                </body>
                </html>
                """)
                review_display.setStyleSheet("""
                    QTextEdit {
                        background: rgba(102, 126, 234, 0.05);
                        border: 1px solid rgba(102, 126, 234, 0.1);
                        border-radius: 8px;
                        padding: 10px;
                        font-size: 12px;
                    }
                """)
                review_display.setMaximumHeight(150)
                card_layout.addWidget(review_display)
                
                self.reviews_layout.addWidget(card)
            
            # Add spacer between products
            spacer = QFrame()
            spacer.setFrameShape(QFrame.Shape.HLine)
            spacer.setStyleSheet("color: rgba(102, 126, 234, 0.2);")
            self.reviews_layout.addWidget(spacer)
        
        # Add final spacer to push cards up
        self.reviews_layout.addStretch()

    def create_results_tab(self):
        """Create results tab."""
        results_tab = QWidget()
        layout = QVBoxLayout(results_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Results Group
        results_group = QGroupBox("📈 Analysis Results")
        results_layout = QVBoxLayout(results_group)
        results_layout.setSpacing(15)

        # Action buttons
        actions_layout = QHBoxLayout()
        
        generate_results_btn = ModernButton("💾 Generate Graphs", primary=True)
        generate_results_btn.clicked.connect(self.generate_results)
        
        view_plots_btn = ModernButton("📊 View Plots", primary=False)
        view_plots_btn.clicked.connect(self.view_plots)
        
        export_btn = ModernButton("📤 Export Results", primary=False)
        export_btn.clicked.connect(self.export_results)
        
        generate_report_btn = ModernButton("📋 Generate Report", primary=False)
        generate_report_btn.clicked.connect(self.generate_report)
        
        actions_layout.addWidget(generate_results_btn)
        actions_layout.addWidget(view_plots_btn)
        actions_layout.addWidget(export_btn)
        actions_layout.addWidget(generate_report_btn)
        actions_layout.addStretch()
        
        results_layout.addLayout(actions_layout)

        # Results Table
        results_label = QLabel("📊 Cluster Characteristics:")
        results_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        results_layout.addWidget(results_label)

        self.results_table = ModernTableWidget()
        results_layout.addWidget(self.results_table)

        # Reviews Section (will be populated when generating results)
        reviews_label = QLabel("📝 Customer Reviews:")
        reviews_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        results_layout.addWidget(reviews_label)

        # Create a scrollable text area for the reviews
        self.reviews_display = QTextEdit()
        self.reviews_display.setReadOnly(True)
        self.reviews_display.setStyleSheet("""
            QTextEdit {
                background: white;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 15px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
                line-height: 1.6;
            }
        """)
        self.reviews_display.setPlaceholderText(
            "Customer reviews will appear here after generating report..."
        )

        # Add a button to load the report
        load_report_btn = ModernButton("📄 Load Review Analysis Report", primary=False)
        load_report_btn.clicked.connect(self.load_review_analysis_report)

        # Create a layout for the button
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(load_report_btn)
        btn_layout.addStretch()

        # Add widgets to the results layout
        results_layout.addLayout(btn_layout)
        results_layout.addWidget(self.reviews_display)

        layout.addWidget(results_group)
        self.tab_widget.addTab(results_tab, "📈 Results")

    def load_review_analysis_report(self):
        """Load and display the review analysis report with professional styling."""
        try:
            report_path = Path("output/gpt4_analysis_report.txt")
            
            if not report_path.exists():
                QMessageBox.warning(self, "File Not Found", "Review analysis report not found.\nPlease generate the report first.")
                return
                
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
                
            # Format the content with professional styling
            html_content = self._format_report_content(report_content)
            
            self.reviews_display.setHtml(html_content)
            self.update_status("Review analysis report loaded")
            self.append_to_console("✓ Review analysis report loaded successfully")
            
        except Exception as e:
            error_msg = f"Error loading review analysis report: {str(e)}"
            self.append_to_console(f"❌ {error_msg}")
            QMessageBox.critical(self, "Error", f"Failed to load review analysis report:\n{str(e)}")

    def _format_report_content(self, content):
        """Convert text report to styled HTML."""
        lines = content.split('\n')
        html_parts = []
        
        # CSS Styles
        css = """
        <style>
            html, body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: #f8f9fa;
                margin: 0;
                padding: 0;
                height: 100%;
                overflow-y: auto;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                padding: 30px;
                margin-top: 20px;
                margin-bottom: 20px;
                min-height: calc(100vh - 40px);
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 3px solid #007bff;
            }
            .header h1 {
                color: #007bff;
                margin: 0;
                font-size: 28px;
                font-weight: 700;
            }
            .header .date {
                color: #6c757d;
                font-size: 14px;
                margin-top: 5px;
            }
            .section {
                margin: 25px 0;
            }
            .section-title {
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white;
                padding: 12px 20px;
                margin: 20px 0 15px 0;
                border-radius: 5px;
                font-weight: 600;
                font-size: 18px;
            }
            .subsection-title {
                color: #495057;
                font-weight: 600;
                font-size: 16px;
                margin: 15px 0 10px 0;
                padding-left: 10px;
                border-left: 4px solid #28a745;
            }
            .bullet-point {
                margin: 8px 0;
                padding-left: 20px;
                position: relative;
            }
            .bullet-point:before {
                content: "•";
                color: #007bff;
                font-weight: bold;
                position: absolute;
                left: 0;
            }
            .highlight-box {
                background: #e7f3ff;
                border: 1px solid #007bff;
                border-radius: 5px;
                padding: 15px;
                margin: 15px 0;
            }
            .data-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .data-table th {
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white;
                padding: 12px 15px;
                text-align: left;
                font-weight: 600;
                font-size: 14px;
            }
            .data-table td {
                padding: 10px 15px;
                border-bottom: 1px solid #dee2e6;
                font-size: 13px;
            }
            .data-table tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            .data-table tr:hover {
                background-color: #e3f2fd;
            }
            .cluster-box {
                background: #f8f9fa;
                border-left: 5px solid #28a745;
                padding: 15px;
                margin: 15px 0;
                border-radius: 0 5px 5px 0;
            }
            .cluster-title {
                color: #28a745;
                font-weight: 600;
                margin-bottom: 10px;
            }
            .metric {
                display: inline-block;
                background: #007bff;
                color: white;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 12px;
                font-weight: 600;
                margin: 2px;
            }
            .separator {
                border-top: 2px solid #dee2e6;
                margin: 30px 0;
            }
        </style>
        """
        
        html_parts.append(f"<html><head>{css}</head><body><div class='container'>")
        
        # Add header
        html_parts.append("""
        <div class='header'>
            <h1>Customer Review Analysis Report</h1>
            <div class='date'>Generated on June 15, 2025</div>
        </div>
        """)
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Main section headers (### format)
            if line.startswith('### '):
                title = line.replace('### ', '').strip()
                html_parts.append(f"<div class='section-title'>{title}</div>")
            
            # Subsection headers (#### format)
            elif line.startswith('#### '):
                title = line.replace('#### ', '').strip()
                html_parts.append(f"<div class='subsection-title'>{title}</div>")
            
            # Bullet points
            elif line.startswith('- '):
                content = line.replace('- ', '').strip()
                html_parts.append(f"<div class='bullet-point'>{content}</div>")
            
            # Table detection (starts with |)
            elif line.startswith('|') and '|' in line:
                table_html = self._parse_table(lines, i)
                html_parts.append(table_html)
                # Skip processed table lines
                while i < len(lines) and lines[i].strip().startswith('|'):
                    i += 1
                continue
            
            # Cluster sections
            elif line.startswith('Cluster ') and ':' in line:
                cluster_content = self._parse_cluster_section(lines, i)
                html_parts.append(cluster_content)
            
            # Quantitative analysis section
            elif 'QUANTITATIVE ANALYSIS' in line:
                html_parts.append("<div class='separator'></div>")
                html_parts.append(f"<div class='section-title'>{line}</div>")
            
            # Key-value pairs (contains :)
            elif ':' in line and not line.startswith('#'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    html_parts.append(f"<div><strong>{key}:</strong> {value}</div>")
                else:
                    html_parts.append(f"<div>{line}</div>")
            
            # Regular paragraphs
            elif line and not line.startswith('='):
                html_parts.append(f"<div style='margin: 10px 0;'>{line}</div>")
            
            # Separators
            elif line.startswith('='):
                html_parts.append("<div class='separator'></div>")
            
            i += 1
        
        html_parts.append("</div></body></html>")
        return ''.join(html_parts)

    def _parse_table(self, lines, start_index):
        """Parse table from text and convert to HTML."""
        table_lines = []
        i = start_index
        
        # Collect all table lines
        while i < len(lines) and lines[i].strip().startswith('|'):
            table_lines.append(lines[i].strip())
            i += 1
        
        if not table_lines:
            return ""
        
        html = "<table class='data-table'>"
        
        # First line is header
        if table_lines:
            header_cells = [cell.strip() for cell in table_lines[0].split('|')[1:-1]]
            html += "<thead><tr>"
            for cell in header_cells:
                html += f"<th>{cell}</th>"
            html += "</tr></thead>"
        
        # Skip separator line (if exists)
        data_start = 2 if len(table_lines) > 1 and '-' in table_lines[1] else 1
        
        # Data rows
        html += "<tbody>"
        for line in table_lines[data_start:]:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            html += "<tr>"
            for cell in cells:
                html += f"<td>{cell}</td>"
            html += "</tr>"
        html += "</tbody></table>"
        
        return html

    def _parse_cluster_section(self, lines, start_index):
        """Parse cluster section and format nicely."""
        line = lines[start_index].strip()
        cluster_title = line
        content = []
        
        i = start_index + 1
        while i < len(lines) and not lines[i].startswith('Cluster ') and not lines[i].startswith('###'):
            if lines[i].strip():
                content.append(lines[i].strip())
            i += 1
        
        html = f"""
        <div class='cluster-box'>
            <div class='cluster-title'>{cluster_title}</div>
            <div>{'<br>'.join(content)}</div>
        </div>
        """
        return html

    def _format_review_analysis(self, text):
        """Convert the raw analysis text to properly formatted HTML with special handling for Customer Reviews"""
        sections = text.split('\n\n')  # Split by double newlines to get sections
        html_sections = []
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # Special handling for Customer Reviews section
            if section.startswith("Customer Reviews:"):
                html_sections.append(self._format_customer_reviews_section(section))
            # Handle tables
            elif '|' in section and ('---' in section or section.count('|') > 2):
                html_sections.append(self._format_table_section(section))
            # Handle headings
            elif section.endswith(':'):
                html_sections.append(f"""
                <div style="
                    background: linear-gradient(90deg, rgba(102,126,234,0.1), rgba(102,126,234,0.3));
                    border-left: 4px solid #667eea;
                    padding: 12px 15px;
                    margin: 20px 0 10px 0;
                    font-size: 16px;
                    font-weight: 600;
                    color: #2d3748;
                ">
                    {section[:-1]}  <!-- Remove trailing colon -->
                </div>
                """)
            # Handle bullet points
            elif section.startswith('- '):
                html_sections.append(f"""
                <div style="
                    margin-left: 20px;
                    color: #4a5568;
                    line-height: 1.6;
                ">
                    • {section[2:]}
                </div>
                """)
            # Regular paragraphs
            else:
                html_sections.append(f"""
                <p style="
                    margin: 10px 0;
                    color: #2d3748;
                    line-height: 1.6;
                ">
                    {section}
                </p>
                """)
        
        return """
        <html>
        <head>
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                color: #2d3748;
                background-color: white;
                padding: 20px;
                line-height: 1.6;
            }
            .review-card {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .review-header {
                font-weight: 600;
                color: #667eea;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
            }
            .review-header-icon {
                margin-right: 8px;
                font-size: 18px;
            }
            .review-content {
                color: #4a5568;
                padding-left: 26px;  /* Align with icon */
            }
            .review-highlight {
                background-color: rgba(102,126,234,0.1);
                padding: 2px 4px;
                border-radius: 4px;
                font-weight: 500;
            }
            .review-metrics {
                margin-top: 10px;
                font-size: 13px;
                color: #718096;
            }
            .review-metrics span {
                display: inline-block;
                margin-right: 15px;
            }
        </style>
        </head>
        <body>
        """ + "".join(html_sections) + """
        </body>
        </html>
        """

    def _format_customer_reviews_section(self, section):
        """Special formatting for the Customer Reviews section"""
        lines = section.split('\n')
        header = lines[0]  # "Customer Reviews:"
        content_lines = lines[1:]
        
        html_content = f"""
        <div style="
            background: linear-gradient(90deg, rgba(102,126,234,0.1), rgba(102,126,234,0.3));
            border-left: 4px solid #667eea;
            padding: 12px 15px;
            margin: 20px 0 10px 0;
            font-size: 16px;
            font-weight: 600;
            color: #2d3748;
        ">
            {header[:-1]}  <!-- Remove trailing colon -->
        </div>
        """
        
        current_product = None
        current_cluster = None
        
        for line in content_lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect product headers (assuming format "Product: XYZ")
            if line.startswith("Product:"):
                current_product = line.split(":")[1].strip()
                html_content += f"""
                <div style="
                    margin: 15px 0 5px 0;
                    font-weight: 600;
                    color: #4a5568;
                    font-size: 15px;
                ">
                    🍏 {current_product}
                </div>
                """
            # Detect cluster headers (assuming format "Cluster X:")
            elif line.startswith("Cluster") and line.endswith(":"):
                current_cluster = line[:-1]  # Remove colon
                html_content += f"""
                <div style="
                    margin: 10px 0 5px 20px;
                    font-weight: 500;
                    color: #667eea;
                    font-size: 14px;
                ">
                    👥 {current_cluster}
                </div>
                """
            # Regular review content
            else:
                # Extract metrics if present (assuming format "Rating: X, Quality: Y")
                metrics = {}
                if "Rating:" in line or "Quality:" in line:
                    metric_parts = re.findall(r'(\w+):\s*([\d\.]+)', line)
                    metrics = {k: v for k, v in metric_parts}
                    # Remove metrics from review text
                    review_text = re.sub(r'\s*(Rating|Quality|Price|Service|Delivery):\s*[\d\.]+', '', line)
                else:
                    review_text = line
                    
                html_content += f"""
                <div class="review-card">
                    <div class="review-header">
                        <span class="review-header-icon">💬</span>
                        <span>Customer Review</span>
                    </div>
                    <div class="review-content">
                        {review_text}
                    </div>
                    {self._format_review_metrics(metrics) if metrics else ''}
                </div>
                """
        
        return html_content

    def _format_review_metrics(self, metrics):
        """Format review metrics into a nice display"""
        if not metrics:
            return ""
            
        metric_items = []
        for name, value in metrics.items():
            metric_items.append(f"""
            <span>
                <span style="font-weight: 500; color: #4a5568;">{name}:</span> 
                <span style="font-weight: 600; color: #2d3748;">{value}</span>
            </span>
            """)
        
        return f"""
        <div class="review-metrics">
            {"".join(metric_items)}
        </div>
        """

    def _format_table_section(self, section):
        """Format a table section"""
        rows = [row.strip() for row in section.split('\n') if row.strip()]
        if len(rows) < 2:  # Need header and at least one row
            return f"<p>{section}</p>"
            
        headers = [h.strip() for h in rows[0].split('|') if h.strip()]
        
        html = """
        <div style="
            margin: 15px 0;
            overflow-x: auto;
        ">
            <table style="
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            ">
                <thead>
                    <tr style="
                        background: linear-gradient(135deg, #667eea, #8a63d2);
                        color: white;
                    ">
        """
        
        for header in headers:
            html += f"""
                        <th style="
                            padding: 12px 15px;
                            text-align: left;
                            font-weight: 600;
                        ">
                            {header}
                        </th>
            """
        
        html += """
                    </tr>
                </thead>
                <tbody>
        """
        
        for row in rows[2:]:  # Skip header and separator
            cells = [c.strip() for c in row.split('|') if c.strip()]
            if len(cells) != len(headers):
                continue
                
            html += """
                    <tr style="
                        border-bottom: 1px solid #e2e8f0;
                    ">
            """
            
            for cell in cells:
                html += f"""
                        <td style="
                            padding: 10px 15px;
                            vertical-align: top;
                        ">
                            {cell}
                        </td>
                """
            
            html += """
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html

    def _format_html_table(self, table_rows):
        """Convert markdown-style table to HTML table with styling."""
        if len(table_rows) < 2:  # Need at least header and separator
            return ""
        
        # Process header
        headers = [h.strip() for h in table_rows[0].split('|') if h.strip()]
        
        # Start building HTML table
        table_html = ['<table class="analysis-table">']
        table_html.append('<thead><tr>')
        for header in headers:
            table_html.append(f'<th>{header}</th>')
        table_html.append('</tr></thead><tbody>')
        
        # Process rows (skip separator line)
        for row in table_rows[2:]:
            cells = [c.strip() for c in row.split('|') if c.strip()]
            if len(cells) != len(headers):
                continue
                
            table_html.append('<tr>')
            for cell in cells:
                # Format numeric cells right-aligned
                if cell.replace('.', '').replace(',', '').isdigit():
                    table_html.append(f'<td style="text-align: right;">{cell}</td>')
                else:
                    table_html.append(f'<td>{cell}</td>')
            table_html.append('</tr>')
        
        table_html.append('</tbody></table>')
        return "".join(table_html)

    def generate_report(self):
        """Generate analysis report using GPT-4 analysis."""
        global global_cluster_reviews, global_cluster_characteristics
        
        if global_cluster_reviews is None or global_cluster_characteristics is None:
            QMessageBox.warning(
                self, 
                "Warning", 
                "No analysis data available!\n\n"
                "Please complete these steps first:\n"
                "1. Load your data\n"
                "2. Run clustering analysis\n"  
                "3. Generate cluster reviews\n"
                "4. Then generate this report"
            )
            return
            
        try:
            self.update_status("Running GPT-4 analysis...")
            self.append_to_console("\n🔍 Starting GPT-4 analysis of reviews...")
            self.append_to_console(f"📊 Analyzing {len(global_cluster_reviews)} products...")
            
            # Create output directory if it doesn't exist
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            # Disable the report button to prevent multiple clicks
            if hasattr(self, 'generate_report_btn'):
                self.generate_report_btn.setEnabled(False)
            
            # Start worker thread for GPT-4 analysis
            self.worker = Worker(
                analyze_reviews_with_gpt4,
                global_cluster_reviews,
                global_cluster_characteristics
            )
            self.worker.finished.connect(self.on_report_generated)
            self.worker.error.connect(self.on_report_error)
            self.worker.start()
            
        except Exception as e:
            error_msg = f"Error starting analysis: {str(e)}"
            self.append_to_console(f"❌ {error_msg}")
            QMessageBox.critical(self, "Error", error_msg)
            
            # Re-enable button on error
            if hasattr(self, 'generate_report_btn'):
                self.generate_report_btn.setEnabled(True)

    def on_report_generated(self, analysis_result):
        """Handle successful report generation"""
        global global_analysis_result
        
        try:
            global_analysis_result = analysis_result
            
            # Display the analysis in the text widget
            if hasattr(self, 'review_analysis_display'):
                self.review_analysis_display.setText(analysis_result)
            elif hasattr(self, 'advanced_results'):
                self.advanced_results.setPlainText(analysis_result)
            
            # Store the analysis result
            self.store_analysis_results(
                global_cluster_reviews, 
                global_cluster_characteristics, 
                analysis_result
            )
            
            # Save to file
            output_dir = Path("output")
            report_path = output_dir / "gpt4_analysis_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(analysis_result)
            
            self.append_to_console("✅ GPT-4 analysis completed successfully!")
            self.append_to_console(f"📄 Report saved: {report_path}")
            self.update_status("GPT-4 analysis completed")
            
            # Show success message
            QMessageBox.information(
                self, 
                "Analysis Complete", 
                f"GPT-4 analysis completed successfully!\n\n"
                f"📄 Report saved to: {report_path}\n"
                f"📊 Results displayed in the interface\n"
                f"🔄 Ready for export to Excel!"
            )
            
        except Exception as e:
            self.on_report_error(f"Error processing analysis result: {str(e)}")
        
        finally:
            # Re-enable the report button
            if hasattr(self, 'generate_report_btn'):
                self.generate_report_btn.setEnabled(True)

    def on_report_error(self, error_message):
        """Handle report generation error"""
        self.append_to_console(f"❌ GPT-4 analysis failed: {error_message}")
        self.update_status("GPT-4 analysis failed")
        
        QMessageBox.critical(
            self, 
            "Analysis Error", 
            f"GPT-4 analysis failed:\n\n{error_message}\n\n"
            f"Please check your API configuration and try again."
        )
        
        # Re-enable the report button
        if hasattr(self, 'generate_report_btn'):
            self.generate_report_btn.setEnabled(True)

    def store_analysis_results(self, cluster_reviews, cluster_characteristics, analysis_result):
        """Store analysis results for export"""
        global global_analysis_result, global_cluster_reviews, global_cluster_characteristics
        
        try:
            # Store in global variables for export
            if cluster_reviews is not None:
                global_cluster_reviews = cluster_reviews
                self.cluster_reviews = cluster_reviews
            
            if cluster_characteristics is not None:
                global_cluster_characteristics = cluster_characteristics
                self.cluster_characteristics = cluster_characteristics
            
            if analysis_result is not None:
                global_analysis_result = analysis_result
                self.analysis_result = analysis_result
            
            # Mark analysis as completed
            self.analysis_completed = True
            
            # Log what was stored
            self.append_to_console("✅ Analysis results stored for export!")
            if cluster_reviews:
                product_count = len(cluster_reviews)
                total_clusters = sum(len(reviews) for reviews in cluster_reviews.values())
                self.append_to_console(f"📊 Stored cluster reviews: {product_count} products, {total_clusters} clusters")
            
            if cluster_characteristics:
                self.append_to_console(f"📈 Stored cluster characteristics: {len(cluster_characteristics)} clusters")
            
            if analysis_result:
                char_count = len(str(analysis_result))
                self.append_to_console(f"📋 Stored GPT-4 analysis: {char_count} characters")
            
        except Exception as e:
            error_msg = f"Error storing results: {str(e)}"
            self.append_to_console(f"❌ {error_msg}")
            print(f"Storage error details: {traceback.format_exc()}")

    # Event handlers
    def on_pipeline_finished(self, result, output):
        """Handle pipeline completion with HTML formatting."""
        # Format the output as HTML
        html_output = self._format_pipeline_output(output)
        
        # Move cursor to end before inserting
        cursor = self.console_output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertHtml(html_output)
        
        if global_cluster_reviews is None or global_analysis_result is None:
            warning_html = """
            <div style="color: #F6AD55; font-weight: bold;">
                ⚠️ Warning: Pipeline completed but some results are missing
            </div>
            """
            cursor.insertHtml(warning_html)
        else:
            success_html = """
            <div style="color: #48BB78; font-weight: bold;">
                ✓ All results generated successfully
            </div>
            """
            cursor.insertHtml(success_html)
        
        # Scroll to bottom
        self.console_output.verticalScrollBar().setValue(
            self.console_output.verticalScrollBar().maximum()
        )
        
        # Update the results table if we have characteristics
        if global_cluster_characteristics is not None:
            if isinstance(global_cluster_characteristics, dict):
                df = pd.DataFrame.from_dict(global_cluster_characteristics, orient='index')
                self.results_table.populate_from_dataframe(df)
            else:
                self.results_table.populate_from_dataframe(global_cluster_characteristics)

    def _format_pipeline_output(self, output):
        """Format pipeline output with HTML styling."""
        sections = []
        current_section = []
        
        # Split output into logical sections
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Detect section headers
            if line.startswith("====") or line.startswith("----"):
                if current_section:
                    sections.append("\n".join(current_section))
                    current_section = []
            else:
                current_section.append(line)
        
        if current_section:
            sections.append("\n".join(current_section))
        
        # Format each section
        html_sections = []
        for section in sections:
            if "Clustering" in section:
                header = "🔍 Customer Clustering"
                color = "#4299E1"  # Blue
            elif "Generating" in section and "reviews" in section:
                header = "📝 Review Generation"
                color = "#9F7AEA"  # Purple
            elif "Analysis" in section:
                header = "📊 Cluster Analysis"
                color = "#48BB78"  # Green
            else:
                header = "Pipeline Output"
                color = "#F6AD55"  # Orange
                
            # Format the section
            html_section = f"""
            <div style="margin-bottom: 15px;">
                <div style="color: {color}; font-weight: bold; font-size: 14px; margin-bottom: 5px;">
                    {header}
                </div>
                <div style="color: #E2E8F0; font-family: 'Consolas', monospace; font-size: 12px;">
                    {section.replace('\n', '<br>')}
                </div>
            </div>
            """
            html_sections.append(html_section)
        
        return "".join(html_sections)

    def enable_results_actions(self, enabled):
        """Enable or disable results-related actions."""
        # Find the generate results button in the results tab
        results_tab = self.tab_widget.widget(2)  # Assuming results tab is index 2
        for child in results_tab.findChildren(QPushButton):
            if "Generate Results" in child.text():
                child.setEnabled(enabled)

    def on_clustering_finished(self, result, output):
        """Handle clustering completion."""
        global global_cluster_characteristics
        
        df_clustered, cluster_characteristics = result
        global_cluster_characteristics = cluster_characteristics
        
        self.append_to_console(output)
        self.update_status("Clustering completed successfully")
        
        # Update results table
        self.results_table.populate_from_dataframe(cluster_characteristics)

    def on_review_generation_finished(self, result, output):
        """Handle review generation completion."""
        global global_cluster_reviews
        
        global_cluster_reviews = result
        
        self.append_to_console(output)
        self.update_status("Review generation completed successfully")

    def on_analysis_only_finished(self, result, output):
        """Handle analysis only completion."""
        global global_df, global_cluster_characteristics
        
        df_clustered, cluster_characteristics = result
        global_df = df_clustered
        global_cluster_characteristics = cluster_characteristics
        
        self.append_to_console(output)
        self.update_status("Analysis completed successfully")
        
        # Optional: Update results table if you want
        if cluster_characteristics is not None:
            # Convert dict to DataFrame if needed
            if isinstance(cluster_characteristics, dict):
                cluster_characteristics = pd.DataFrame.from_dict(cluster_characteristics, orient='index')
            self.results_table.populate_from_dataframe(cluster_characteristics)

    def on_advanced_analysis_finished(self, result, output):
        """Handle advanced analysis completion."""
        self.append_to_console(output)
        
        # Check which type of analysis this is
        if isinstance(result, dict) and 'metrics' in result:
            self.display_product_performance(result)
        elif "Cluster Comparison" in output:
            self.display_cluster_comparison(output)
        else:
            # For other analysis types, format with tables
            formatted_output = self._format_analysis_text(filter_progress_bar_output(output))
            self.advanced_results.setHtml(f"""
            <html>
            <body style="font-family: 'Segoe UI', Arial, sans-serif; color: #2d3748;">
                {formatted_output}
            </body>
            </html>
            """)
        
        self.update_status("Advanced analysis completed")

    def on_pipeline_error(self, error_msg):
        """Handle pipeline error with HTML formatting."""
        error_html = f"""
        <div style="color: #F56565; font-weight: bold;">
            ❌ Pipeline Error: {error_msg}
        </div>
        """
        
        cursor = self.console_output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertHtml(error_html)
        
        self.update_status("Pipeline failed")
        QMessageBox.critical(self, "Pipeline Error", f"Pipeline failed with error:\n{error_msg}")

    def on_analysis_error(self, error_msg):
        """Handle analysis error."""
        self.append_to_console(f"❌ Analysis Error: {error_msg}")
        self.update_status("Analysis failed")
        QMessageBox.critical(self, "Analysis Error", f"Analysis failed with error:\n{error_msg}")

    def display_product_performance(self, performance_data):
        """Display product performance analysis with visual elements"""
        if not performance_data:
            return
            
        # Clear previous content
        self.advanced_results.clear()
        
        # Create HTML content with styling
        html_content = """
        <html>
        <head>
        <style>
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                color: #2d3748;
                background-color: #f8f9fa;
                padding: 15px;
            }
            h1 { 
                color: #2d3748; 
                font-size: 22px;
                font-weight: 700;
                margin-bottom: 25px;
                padding-bottom: 12px;
                border-bottom: 2px solid rgba(102, 126, 234, 0.3);
            }
            
            /* Product Cards Styling */
            .product-card { 
                background: white; 
                border: 1px solid #e2e8f0; 
                border-radius: 12px; 
                padding: 20px; 
                margin-bottom: 25px;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.08);
            }
            .product-name { 
                font-weight: 700; 
                color: #2d3748;
                font-size: 18px;
                margin-bottom: 18px;
            }
            
            /* Metrics Styling - UPDATED TEXT COLOR TO BLACK */
            .metric { 
                display: flex; 
                align-items: center; 
                margin-bottom: 12px;
            }
            .metric-name { 
                width: 200px; 
                font-weight: 600;
                color: #4a5568;
                font-size: 14px;
            }
            .rating-bar-container { 
                flex-grow: 1; 
                height: 24px; 
                background: #edf2f7; 
                border-radius: 12px;
                overflow: hidden;
                position: relative;
            }
            .rating-bar { 
                height: 100%; 
                background: linear-gradient(90deg, #667eea, #8a63d2);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding-right: 10px;
                color: #000000; /* Changed from white to black */
                font-size: 12px;
                font-weight: 700;
                min-width: 40px;
                text-shadow: 0 1px 1px rgba(255,255,255,0.5); /* Added light text shadow for contrast */
            }
            
            /* Analysis Section Styling */
            .analysis-section { 
                background: white; 
                padding: 25px; 
                border-radius: 12px; 
                margin-top: 30px;
                box-shadow: 0 4px 16px rgba(102, 126, 234, 0.1);
            }
            
            /* Table Styling */
            .analysis-table-container {
                margin: 20px 0;
                overflow-x: auto;
            }
            .analysis-table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                font-size: 14px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .analysis-table th {
                background: linear-gradient(135deg, #667eea, #8a63d2);
                color: white;
                text-align: left;
                padding: 12px 15px;
                font-weight: 600;
            }
            .analysis-table td {
                padding: 10px 15px;
                border-bottom: 1px solid #e2e8f0;
                vertical-align: top;
            }
            .analysis-table tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            .analysis-table tr:hover {
                background-color: rgba(102, 126, 234, 0.05);
            }
            
            /* Key Points Styling */
            .key-point {
                background: rgba(102, 126, 234, 0.08);
                border-left: 3px solid #667eea;
                padding: 12px 15px;
                margin: 15px 0;
                border-radius: 0 6px 6px 0;
            }
        </style>
        </head>
        <body>
        <h1>📊 Product Performance Dashboard</h1>
        """
        
        # Add product metrics
        metrics = performance_data.get("metrics", {})
        for product, ratings in metrics.items():
            html_content += f"""
            <div class="product-card">
                <div class="product-name">{product}</div>
            """
            
            for metric, score in ratings.items():
                width = min(100, score * 20)  # Scale 1-5 rating to 20-100%
                html_content += f"""
                <div class="metric">
                    <div class="metric-name">{metric.replace('_', ' ').title()}:</div>
                    <div class="rating-bar-container">
                        <div class="rating-bar" style="width: {width}%">{score:.1f}/5</div>
                    </div>
                </div>
                """
            
            html_content += "</div>"
        
        # Add analysis text with enhanced formatting
        analysis = performance_data.get("analysis", "")
        if analysis:
            # Process the analysis text to add HTML formatting
            formatted_analysis = self._format_analysis_text(analysis)
            
            html_content += f"""
            <div class="analysis-section">
                <h2>Strategic Insights</h2>
                {formatted_analysis}
            </div>
            """
        
        html_content += "</body></html>"
        
        # Set HTML content
        self.advanced_results.setHtml(html_content)

    def display_cluster_comparison(self, comparison_data):
        """Display cluster comparison analysis with proper tables"""
        self.advanced_results.clear()
        
        html_content = """
        <html>
        <head>
        <style>
            /* Include all the same styles as in display_product_performance */
            body { font-family: 'Segoe UI', Arial, sans-serif; color: #2d3748; }
            h1 { color: #2d3748; font-size: 22px; font-weight: 700; }
            .analysis-section { background: white; padding: 25px; border-radius: 12px; margin: 20px 0; }
            .analysis-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
            .analysis-table th { background: #667eea; color: white; padding: 12px; text-align: left; }
            .analysis-table td { padding: 10px 15px; border-bottom: 1px solid #e2e8f0; }
            .analysis-table tr:nth-child(even) { background-color: #f8f9fa; }
        </style>
        </head>
        <body>
        <h1>🔍 Cluster Comparison Analysis</h1>
        """
        
        if isinstance(comparison_data, str):
            # Format the text with tables
            html_content += self._format_analysis_text(comparison_data)
        else:
            # Handle structured data if available
            html_content += "<div class='analysis-section'>"
            html_content += "<p>No structured comparison data available.</p>"
            html_content += "</div>"
        
        html_content += "</body></html>"
        self.advanced_results.setHtml(html_content)

    def _format_analysis_text(self, analysis_text):
        """Format the raw analysis text into structured HTML with proper tables"""
        # Split into paragraphs
        paragraphs = [p.strip() for p in analysis_text.split('\n') if p.strip()]
        
        formatted_html = ""
        in_table = False
        table_rows = []
        
        for para in paragraphs:
            # Detect tables (lines with multiple | characters)
            if '|' in para and ('---' in para or para.count('|') > 2):
                if not in_table:
                    in_table = True
                    table_rows = []
                table_rows.append(para)
                continue
            elif in_table:
                # Process the collected table rows
                formatted_html += self._format_table(table_rows)
                in_table = False
                table_rows = []
            
            # Detect numbered points (1., 2., etc.)
            if re.match(r'^\d+\.', para):
                formatted_html += f'<div class="key-point"><div class="key-point-title">{para.split(".", 1)[0]}.</div>{para.split(".", 1)[1]}</div>'
            # Detect headings
            elif para.endswith(':'):
                formatted_html += f'<p><strong>{para}</strong></p>'
            else:
                formatted_html += f'<p>{para}</p>'
        
        # Process any remaining table at the end
        if in_table:
            formatted_html += self._format_table(table_rows)
        
        return formatted_html

    def _format_table(self, table_rows):
        """Convert markdown-style table to HTML with proper styling"""
        if not table_rows or len(table_rows) < 2:
            return ""
        
        # Extract headers
        headers = [h.strip() for h in table_rows[0].split('|') if h.strip()]
        
        # Start building HTML table
        table_html = """
        <div class="analysis-table-container">
        <table class="analysis-table">
            <thead>
                <tr>
        """
        
        # Add headers
        for header in headers:
            table_html += f'<th>{header}</th>'
        
        table_html += """
                </tr>
            </thead>
            <tbody>
        """
        
        # Add rows (skip the separator line)
        for row in table_rows[2:]:
            cells = [c.strip() for c in row.split('|') if c.strip()]
            if len(cells) != len(headers):
                continue
                
            table_html += '<tr>'
            for cell in cells:
                table_html += f'<td>{cell}</td>'
            table_html += '</tr>'
        
        table_html += """
            </tbody>
        </table>
        </div>
        """
        
        return table_html
    


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Customer Analysis Suite")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Apple Analysis")
    
    # Apply application-wide styling
    app.setStyleSheet("""
        QApplication {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QToolTip {
            background-color: #2d3748;
            color: white;
            border: 1px solid #4a5568;
            border-radius: 4px;
            padding: 8px;
            font-size: 11px;
        }
    """)
    
    # Create and show main window
    window = CustomerAnalysisGUI()
    window.show()
    
    # Center window on screen
    screen = QApplication.primaryScreen().geometry()
    window.move(
        (screen.width() - window.width()) // 2,
        (screen.height() - window.height()) // 2
    )
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
