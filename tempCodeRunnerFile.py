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