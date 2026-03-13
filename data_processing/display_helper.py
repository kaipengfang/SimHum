"""
Display Helper Module for Unified Data Converter

Handles all Rich-based terminal output and progress display functionality.
Separates display logic from core data processing logic.
"""

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from tqdm import tqdm
    RICH_AVAILABLE = True
except ImportError:
    print("Warning: rich library not available. Install with: pip install rich")
    RICH_AVAILABLE = False


class DisplayHelper:
    """Helper class for all display and progress output"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.rich_available = RICH_AVAILABLE
    
    def print_header(self):
        """Print conversion header"""
        if self.rich_available:
            self.console.print(Panel.fit(
                "[bold blue]Unified Data Converter[/bold blue]\n"
                "Converting robotics datasets to unified format",
                style="blue"
            ))
        else:
            print("\n" + "="*60)
            print("UNIFIED DATA CONVERTER")  
            print("="*60)
    
    def print_data_source_summary(self, processors):
        """Print beautiful summary of data sources"""
        if not self.rich_available:
            # Fallback to simple print
            print("\n" + "="*50)
            print("DATA SOURCES SUMMARY")
            print("="*50)
            for processor in processors:
                info = processor.get_data_info()
                print(f"{info.data_type}: {info.total_episodes} episodes, {info.num_tasks} tasks")
            print("="*50)
            return
        
        # Rich table output
        table = Table(title="Data Sources Summary", show_header=True, header_style="bold magenta")
        table.add_column("Data Type", style="cyan", no_wrap=True)
        table.add_column("Tasks", style="green")
        table.add_column("Episodes", justify="right", style="yellow")
        table.add_column("Dimension", justify="center", style="red")
        table.add_column("Has Video", justify="center", style="blue")
        
        total_episodes = 0
        for processor in processors:
            info = processor.get_data_info()
            total_episodes += info.total_episodes
            
            # Format task names for display
            task_display = ", ".join(info.task_names[:2])  # Show first 2 tasks
            if len(info.task_names) > 2:
                task_display += f" (+{len(info.task_names)-2} more)"
            
            table.add_row(
                info.data_type,
                task_display,
                str(info.total_episodes),
                f"{info.output_dimension}D",
                "✓" if info.has_video else "✗"
            )
        
        # Add summary row
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{len(processors)} sources[/bold]", 
            f"[bold]{total_episodes}[/bold]",
            "",
            ""
        )
        
        self.console.print(table)
        self.console.print()
    
    def print_target_dimension(self, target_dim, robot_dim, human_dim):
        """Print target dimension information"""
        if self.rich_available:
            dim_style = "green" if target_dim == robot_dim else "yellow"
            self.console.print(f"🎯 Target dimension: [{dim_style}]{target_dim}D[/{dim_style}]")
            if target_dim == human_dim:
                self.console.print("   ℹ️  Robot data will be zero-padded to 46D for consistency")
            self.console.print()
        else:
            print(f"🎯 Target dimension: {target_dim}D")
            if target_dim == human_dim:
                print("   Robot data will be zero-padded to 46D for consistency")
    
    def print_output_directory(self, output_path):
        """Print output directory path"""
        if self.rich_available:
            self.console.print(f"📁 Output directory: [cyan]{output_path}[/cyan]")
            self.console.print()
        else:
            print(f"📁 Output directory: {output_path}")
    
    def print_coordinate_alignment(self, coord_aligned_sources):
        """Print coordinate alignment information"""
        if coord_aligned_sources:
            if self.rich_available:
                self.console.print(f"\n[bold yellow]📍 Coordinate Alignment Active:[/bold yellow]")
                self.console.print(f"   {len(coord_aligned_sources)} data source(s) will be transformed to Human coordinate system")
            else:
                print(f"\n📍 Coordinate Alignment Active:")
                print(f"   {len(coord_aligned_sources)} data source(s) will be transformed to Human coordinate system")
    
    def print_instruction_processing(self, num_instructions):
        """Print instruction processing status"""
        if self.rich_available:
            self.console.print("🔤 Processing task instructions and generating embeddings...")
            self.console.print(f"   Generating embeddings for {num_instructions} tasks...")
        else:
            print("🔤 Processing task instructions and generating embeddings...")
            print(f"   Generating embeddings for {num_instructions} tasks...")
    
    def print_instruction_complete(self, num_instructions):
        """Print instruction processing completion"""
        if self.rich_available:
            self.console.print(f"   ✅ Saved embeddings and mappings for {num_instructions} tasks")
        else:
            print(f"   ✅ Saved embeddings and mappings for {num_instructions} tasks")
    
    def print_data_processing_start(self):
        """Print data processing start message"""
        if self.rich_available:
            self.console.print("🤖 Processing episodes and collecting data...")
        else:
            print("🤖 Processing episodes and collecting data...")
    
    def print_processor_info(self, data_type, num_episodes):
        """Print processor processing information"""
        if self.rich_available:
            self.console.print(f"   Processing {data_type}: {num_episodes} episodes")
        else:
            print(f"   Processing {data_type}: {num_episodes} episodes")
    
    def print_coordinate_transform_note(self):
        """Print coordinate transformation note"""
        if self.rich_available:
            self.console.print("   [yellow]Note: Coordinate transformations will be applied during processing[/yellow]")
        else:
            print("   Note: Coordinate transformations will be applied during processing")
    
    def print_saving_data(self):
        """Print saving data message"""
        if self.rich_available:
            self.console.print("💾 Saving processed data...")
        else:
            print("💾 Saving processed data...")
    
    def print_final_summary(self, output_path, total_episodes, total_samples, processing_time, target_dim):
        """Print final processing summary"""
        if self.rich_available:
            # Create summary table
            summary_table = Table(title="Processing Summary", show_header=True, header_style="bold green")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="yellow", justify="right")
            
            summary_table.add_row("Total Episodes", str(total_episodes))
            summary_table.add_row("Total Samples", str(total_samples))
            summary_table.add_row("Output Dimension", f"{target_dim}D")
            summary_table.add_row("Processing Time", f"{processing_time:.1f}s")
            
            self.console.print()
            self.console.print(summary_table)
            
            # Success message
            self.console.print(Panel.fit(
                f"✅ [bold green]Conversion Complete![/bold green]\n\n"
                f"📁 Output saved to: [cyan]{output_path}[/cyan]\n"
                f"📊 Processed [yellow]{total_episodes}[/yellow] episodes successfully\n"
                f"🎯 Final dataset: [green]{target_dim}D[/green] with [yellow]{total_samples}[/yellow] samples",
                style="green"
            ))
        else:
            # Fallback simple output
            print("\n" + "="*50)
            print("PROCESSING SUMMARY")
            print("="*50)
            print(f"Total Episodes: {total_episodes}")
            print(f"Total Samples: {total_samples}")
            print(f"Output Dimension: {target_dim}D")
            print(f"Processing Time: {processing_time:.1f}s")
            print(f"\nOutput saved to: {output_path}")
            print("="*50)
    
    def print_error(self, error_msg):
        """Print error message"""
        if self.rich_available:
            self.console.print(f"   ❌ Error: {error_msg}")
        else:
            print(f"   ❌ Error: {error_msg}")
    
    def create_progress_bar(self, episodes, desc):
        """Create progress bar for episode processing"""
        return tqdm(episodes, desc=desc, disable=not self.rich_available)