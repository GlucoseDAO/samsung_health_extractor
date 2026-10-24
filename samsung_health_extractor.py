"""Samsung Health Data Extractor.

A script to generate CSV files from Samsung Health data exports.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import polars as pl
import typer
from eliot import add_destinations, log_message, start_action

app = typer.Typer(help="Samsung Health Data Extractor")


def setup_logging(log_file: Path | None = None) -> TextIO:
    """Setup Eliot logging to a file with readable formatting.
    
    Args:
        log_file: Path to log file. If None, creates a timestamped log file.
        
    Returns:
        File handle for the log file
    """
    if log_file is None:
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"samsung_health_extract_{timestamp}.log"
    
    # Open log file
    log_handle = log_file.open("w", encoding="utf-8")
    
    # Add file destination - Eliot will write JSON to this file
    # Eliot expects a callable that takes a message dict
    def write_log(message: dict) -> None:
        log_handle.write(json.dumps(message) + "\n")
        log_handle.flush()
    
    add_destinations(write_log)
    
    # Log startup message
    log_message(
        message_type="logging_initialized",
        log_file=str(log_file),
        timestamp=datetime.now().isoformat()
    )
    
    return log_handle


def read_ignore_list(filename: Path = Path("ignore_csvs.txt")) -> set[str]:
    """Read CSV names to ignore from an external file.
    
    Args:
        filename: Path to the file containing CSV names to ignore
        
    Returns:
        Set of cleaned CSV names to ignore
    """
    with start_action(action_type="read_ignore_list", filename=str(filename)) as action:
        ignore_set: set[str] = set()
        
        if not filename.exists():
            action.log(message_type="warning", warning=f"{filename} not found. No files will be ignored.")
            return ignore_set
        
        with filename.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#"):
                    ignore_set.add(line)
        
        if ignore_set:
            action.log(message_type="loaded_ignore_list", count=len(ignore_set), items=sorted(ignore_set))
        else:
            action.log(message_type="info", info="No entries found in ignore file")
        
        return ignore_set


def read_combination_config(filename: Path = Path("data_combination.json")) -> dict[str, Any]:
    """Read data combination configuration from JSON file.
    
    Args:
        filename: Path to the JSON configuration file
        
    Returns:
        Dictionary with combination configurations
    """
    with start_action(action_type="read_combination_config", filename=str(filename)) as action:
        config: dict[str, Any] = {}
        
        if not filename.exists():
            action.log(message_type="warning", warning=f"{filename} not found. No data combinations will be processed.")
            return config
        
        with filename.open("r", encoding="utf-8") as f:
            data = json.load(f)
            config = data.get("data_combinations", {})
        
        if config:
            action.log(message_type="loaded_config", combinations_count=len(config))
            for combo_name, combo_config in config.items():
                output_file = combo_config.get("output_file", "unknown.csv")
                sources_count = len(combo_config.get("sources", []))
                action.log(
                    message_type="combination_info",
                    name=combo_name,
                    output=output_file,
                    sources=sources_count
                )
        else:
            action.log(message_type="info", info="No data combination configurations found")
        
        return config


def clean_csv_name(name: str) -> str:
    """Clean a single CSV name.
    
    Removes .csv extension, 'health' prefix, and trailing numbers.
    
    Args:
        name: Original CSV filename
        
    Returns:
        Cleaned name without extension, prefix, and trailing numbers
    """
    # Remove .csv extension
    name_without_ext = name.replace(".csv", "")
    
    # Find 'health' in the name and remove everything before it including 'health'
    health_index = name_without_ext.find("health")
    if health_index != -1:
        # Remove everything before and including 'health'
        name_without_ext = name_without_ext[health_index + 6:]  # +6 to skip 'health'
    
    # Remove trailing numbers using regex
    cleaned_name = re.sub(r"\d+$", "", name_without_ext)
    
    # Remove any trailing dots or underscores that might be left
    cleaned_name = cleaned_name.rstrip("._")
    
    return cleaned_name


def csv_has_data(file_path: Path, min_rows: int = 10) -> bool:
    """Check if a CSV file contains meaningful data beyond just headers.
    
    Files with min_rows or fewer rows (besides header) are considered to have no data.
    
    Args:
        file_path: Path to the CSV file
        min_rows: Minimum number of data rows required (default: 10)
        
    Returns:
        True if the file has meaningful data, False if it's empty or has minimal data
    """
    with start_action(action_type="check_csv_data", file=str(file_path)) as action:
        try:
            # Read CSV with polars, skipping the first row (which is usually metadata)
            # truncate_ragged_lines handles CSV files with inconsistent column counts
            df = pl.read_csv(
                file_path,
                skip_rows=1,
                infer_schema_length=1000,
                truncate_ragged_lines=True,
                ignore_errors=True
            )
            
            # Check if dataframe is empty
            if df.is_empty():
                action.log(message_type="empty_dataframe")
                return False
            
            # Check if all rows are null
            if df.null_count().sum_horizontal()[0] == df.height * df.width:
                action.log(message_type="all_null")
                return False
            
            # Remove rows that are entirely null
            df_clean = df.filter(~pl.all_horizontal(pl.all().is_null()))
            if df_clean.is_empty():
                action.log(message_type="empty_after_cleaning")
                return False
            
            # Check if we have enough rows for meaningful data
            if len(df_clean) <= min_rows:
                action.log(message_type="insufficient_rows", rows=len(df_clean), required=min_rows)
                return False
            
            # Check if we have at least one non-empty value in the dataframe
            has_data = False
            for col in df_clean.columns:
                # Check for non-null values
                non_null_count = df_clean[col].null_count()
                if non_null_count < len(df_clean):
                    has_data = True
                    break
            
            action.log(message_type="has_data", result=has_data)
            return has_data
            
        except Exception as e:
            action.log(message_type="error", error=str(e))
            return False


def get_cleaned_csv_names(
    csv_paths: list[Path],
    filter_ignored: bool = True,
    ignore_file: Path = Path("ignore_csvs.txt"),
    hierarchical: bool = True,
    filter_empty: bool = True,
    filter_config: bool = True,
    config_file: Path = Path("data_combination.json"),
) -> dict[str, dict[str, list[str]]] | list[str]:
    """Extract CSV file names and clean them by removing prefixes and suffixes.
    
    Args:
        csv_paths: List of paths to CSV files
        filter_ignored: Whether to filter out ignored CSV names
        ignore_file: Path to file containing CSV names to ignore
        hierarchical: Whether to return hierarchical structure or flat list
        filter_empty: Whether to filter out CSV files with no data
        filter_config: Whether to only process CSV files in configuration
        config_file: Path to data combination JSON file
        
    Returns:
        If hierarchical=True: dict with categories and subcategories
        If hierarchical=False: list of cleaned CSV names
    """
    with start_action(
        action_type="get_cleaned_csv_names",
        filter_ignored=filter_ignored,
        hierarchical=hierarchical,
        filter_empty=filter_empty,
        filter_config=filter_config
    ) as action:
        # Get ignore list if filtering is enabled
        ignore_set: set[str] = set()
        if filter_ignored:
            ignore_set = read_ignore_list(ignore_file)
        
        # Get CSV configuration if filtering is enabled
        enabled_csvs: set[str] = set()
        if filter_config:
            combination_config = read_combination_config(config_file)
            csv_filtering = combination_config.get("csv_filtering", {})
            enabled_csvs = set(csv_filtering.get("enabled_csvs", []))
            if not enabled_csvs:
                action.log(message_type="info", info="No enabled CSV files found in configuration. All files will be processed.")
            else:
                action.log(message_type="filtering_enabled", count=len(enabled_csvs), items=sorted(enabled_csvs))
        
        if not hierarchical:
            # Return flat list (original behavior)
            cleaned_names: list[str] = []
            for file_path in csv_paths:
                name = file_path.name
                cleaned_name = clean_csv_name(name)
                
                # Filter out ignored names if filtering is enabled
                if filter_ignored and cleaned_name in ignore_set:
                    action.log(message_type="filtered_ignored", name=cleaned_name)
                    continue
                
                # Filter out empty files if filtering is enabled
                if filter_empty and not csv_has_data(file_path):
                    action.log(message_type="filtered_empty", name=cleaned_name)
                    continue
                
                # Filter out files not in configuration if filtering is enabled
                if filter_config and enabled_csvs and cleaned_name not in enabled_csvs:
                    action.log(message_type="filtered_not_in_config", name=cleaned_name)
                    continue
                
                cleaned_names.append(cleaned_name)
            
            return cleaned_names
        else:
            # Return hierarchical structure
            categories: dict[str, dict[str, list[str]]] = {}
            
            for file_path in csv_paths:
                name = file_path.name
                cleaned_name = clean_csv_name(name)
                
                # Filter out ignored names if filtering is enabled
                if filter_ignored and cleaned_name in ignore_set:
                    action.log(message_type="filtered_ignored", name=cleaned_name)
                    continue
                
                # Filter out empty files if filtering is enabled
                if filter_empty and not csv_has_data(file_path):
                    action.log(message_type="filtered_empty", name=cleaned_name)
                    continue
                
                # Filter out files not in configuration if filtering is enabled
                if filter_config and enabled_csvs and cleaned_name not in enabled_csvs:
                    action.log(message_type="filtered_not_in_config", name=cleaned_name)
                    continue
                
                # Split by dots to get category and subcategory
                parts = cleaned_name.split(".")
                
                if len(parts) == 1:
                    # Main category (no subcategory)
                    category = parts[0]
                    if category not in categories:
                        categories[category] = {"subcategories": [], "files": []}
                    categories[category]["files"].append(cleaned_name)
                elif len(parts) == 2:
                    # Has one dot - treat as main category with no subcategory
                    category = parts[0]
                    if category not in categories:
                        categories[category] = {"subcategories": [], "files": []}
                    categories[category]["files"].append(cleaned_name)
                else:
                    # Has 2+ dots - use second dot as separator for subcategories
                    category = ".".join(parts[:2])  # First two parts as category
                    subcategory = ".".join(parts[2:])  # Everything after second dot as subcategory
                    
                    if category not in categories:
                        categories[category] = {"subcategories": [], "files": []}
                    
                    # Add to subcategories if not already there
                    if subcategory not in categories[category]["subcategories"]:
                        categories[category]["subcategories"].append(subcategory)
            
            # Sort categories and subcategories for consistent output
            for category in categories:
                categories[category]["subcategories"].sort()
                categories[category]["files"].sort()
            
            return dict(sorted(categories.items()))


def process_data_combinations(
    combination_config: dict[str, Any],
    categories: dict[str, dict[str, list[str]]],
    csv_paths: list[Path],
) -> None:
    """Process data combinations according to the configuration.
    
    Combines data from multiple CSV sources and outputs to specified files.
    
    Args:
        combination_config: Configuration dictionary for data combinations
        categories: Hierarchical structure of CSV categories
        csv_paths: List of paths to all CSV files
    """
    with start_action(action_type="process_data_combinations", combinations_count=len(combination_config)) as action:
        # DEBUG: Show all CSV files available for processing
        action.log(
            message_type="debug_available_csvs", 
            count=len(csv_paths), 
            files=[str(p) for p in csv_paths]
        )
        
        # DEBUG: Show cleaned names mapping
        cleaned_names_map = {}
        for file_path in csv_paths:
            cleaned_name = clean_csv_name(file_path.name)
            cleaned_names_map[cleaned_name] = str(file_path)
        action.log(
            message_type="debug_cleaned_names_map",
            mapping=cleaned_names_map
        )
        for combo_name, combo_config in combination_config.items():
            output_file = combo_config.get("output_file", "unknown.csv")
            action.log(message_type="processing_combination", name=combo_name, output=output_file)
            
            try:
                # Get sources for this combination
                sources = combo_config.get("sources", [])
                if not sources:
                    action.log(message_type="no_sources", name=combo_name)
                    continue
                
                # Sort sources by priority
                sources.sort(key=lambda x: x.get("priority", 999))
                
                combined_df: pl.DataFrame | None = None
                merge_key: str | None = None
                
                for i, source in enumerate(sources):
                    csv_name = source.get("csv_name", "")
                    priority = source.get("priority", i + 1)
                    required = source.get("required", False)
                    columns_to_include = source.get("columns_to_include", [])
                    rename_columns = source.get("rename_columns", {})
                    
                    # DEBUG: Show full source configuration
                    action.log(
                        message_type="debug_processing_source", 
                        priority=priority, 
                        csv_name=csv_name,
                        required=required,
                        columns_to_include=columns_to_include,
                        rename_columns=rename_columns,
                        source_config=source
                    )
                    
                    # Find the CSV file path
                    csv_file_path: Path | None = None
                    for file_path in csv_paths:
                        name = file_path.name
                        cleaned_name = clean_csv_name(name)
                        if cleaned_name == csv_name:
                            csv_file_path = file_path
                            break
                    
                    # DEBUG: Show file search result
                    if csv_file_path:
                        action.log(
                            message_type="debug_file_found",
                            csv_name=csv_name,
                            file_path=str(csv_file_path)
                        )
                    else:
                        action.log(
                            message_type="debug_file_not_found",
                            csv_name=csv_name,
                            required=required
                        )
                    
                    if not csv_file_path:
                        if required:
                            action.log(message_type="required_source_not_found", csv_name=csv_name)
                            break
                        else:
                            action.log(message_type="optional_source_not_found", csv_name=csv_name)
                            continue
                    
                    # Read the CSV data
                    try:
                        df = pl.read_csv(
                            csv_file_path,
                            skip_rows=1,
                            infer_schema_length=1000,
                            truncate_ragged_lines=True,
                            ignore_errors=True
                        )
                        
                        # DEBUG: Show raw data loaded
                        action.log(
                            message_type="debug_raw_data_loaded",
                            csv_name=csv_name,
                            rows=len(df),
                            columns=df.columns,
                            dtypes=[str(dtype) for dtype in df.dtypes],
                            sample_head=df.head(3).to_dicts() if len(df) > 0 else []
                        )
                        
                        # Handle merge key rename for standardization
                        source_merge_key = source.get("merge_key", "day_time")
                        merge_key_rename = source.get("merge_key_rename", source_merge_key)
                        
                        # Set merge key from first source, or use the renamed key for standardization
                        if merge_key is None:
                            merge_key = merge_key_rename
                        
                        # Ensure merge key is always included in columns_to_include
                        columns_to_include_with_key = columns_to_include.copy() if columns_to_include else []
                        if source_merge_key not in columns_to_include_with_key and source_merge_key in df.columns:
                            columns_to_include_with_key.append(source_merge_key)
                        
                        # Select only required columns (including merge key)
                        if columns_to_include_with_key:
                            available_columns = [col for col in columns_to_include_with_key if col in df.columns]
                            if not available_columns:
                                action.log(
                                    message_type="no_required_columns",
                                    required=columns_to_include_with_key,
                                    available=df.columns
                                )
                                if required:
                                    break
                                else:
                                    continue
                            
                            # DEBUG: Show column selection
                            action.log(
                                message_type="debug_column_selection",
                                requested=columns_to_include_with_key,
                                available=available_columns,
                                before_columns=df.columns
                            )
                            
                            df = df.select(available_columns)
                            
                            # DEBUG: Show after column selection
                            action.log(
                                message_type="debug_after_column_selection",
                                after_columns=df.columns,
                                sample=df.head(3).to_dicts() if len(df) > 0 else []
                            )
                        
                        # Save the original merge key name before any transformations
                        original_merge_key = source_merge_key
                        
                        # Handle case where merge key needs to be both renamed and kept with another name
                        # If the merge key is also in rename_columns, duplicate it first
                        if rename_columns and source_merge_key in rename_columns:
                            action.log(
                                message_type="debug_duplicating_merge_key_column",
                                merge_key=source_merge_key,
                                target_name=rename_columns[source_merge_key]
                            )
                            # Duplicate the merge key column before any renaming
                            df = df.with_columns(pl.col(source_merge_key).alias(rename_columns[source_merge_key]))
                            action.log(
                                message_type="debug_after_duplication",
                                columns=df.columns
                            )
                        
                        # Rename merge key to standardize across sources
                        if source_merge_key in df.columns and merge_key_rename != source_merge_key:
                            action.log(
                                message_type="debug_renaming_merge_key",
                                from_key=source_merge_key,
                                to_key=merge_key_rename
                            )
                            df = df.rename({source_merge_key: merge_key_rename})
                            source_merge_key = merge_key_rename
                        
                        # Rename other columns if specified (excluding merge key which was already handled)
                        # We need to filter out the ORIGINAL merge key name since it's been renamed/duplicated
                        if rename_columns:
                            # Filter out the original merge key from rename_columns since it's already been duplicated
                            filtered_rename_columns = {
                                k: v for k, v in rename_columns.items() 
                                if k != original_merge_key  # Use original name, not the renamed one
                            }
                            
                            if filtered_rename_columns:
                                action.log(
                                    message_type="debug_renaming_other_columns",
                                    rename_map=filtered_rename_columns,
                                    before_columns=df.columns
                                )
                                df = df.rename(filtered_rename_columns)
                                action.log(
                                    message_type="debug_after_rename",
                                    after_columns=df.columns,
                                    sample=df.head(3).to_dicts() if len(df) > 0 else []
                                )
                        
                        # Convert merge key to date or datetime for proper merging
                        # If merge_key_rename is "datetime", preserve full datetime; otherwise convert to date
                        if merge_key in df.columns:
                            try:
                                use_datetime = (merge_key_rename == "datetime")
                                # Check if column is numeric (milliseconds timestamp)
                                if df[merge_key].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                                    if use_datetime:
                                        df = df.with_columns(
                                            pl.from_epoch(pl.col(merge_key), time_unit="ms").alias(merge_key)
                                        )
                                    else:
                                        df = df.with_columns(
                                            pl.from_epoch(pl.col(merge_key), time_unit="ms").dt.date().alias(merge_key)
                                        )
                                else:
                                    # Try to parse as datetime string
                                    if use_datetime:
                                        df = df.with_columns(
                                            pl.col(merge_key).str.to_datetime()
                                        )
                                    else:
                                        df = df.with_columns(
                                            pl.col(merge_key).str.to_datetime().dt.date()
                                        )
                            except Exception as e:
                                action.log(message_type="datetime_conversion_error", error=str(e))
                        
                        action.log(message_type="loaded_data", rows=len(df), columns=df.columns)
                        
                        # Combine with previous data
                        if combined_df is None:
                            combined_df = df
                            action.log(
                                message_type="debug_base_dataframe_set", 
                                csv_name=csv_name,
                                rows=len(combined_df),
                                columns=combined_df.columns,
                                sample=combined_df.head(3).to_dicts() if len(combined_df) > 0 else []
                            )
                        else:
                            # DEBUG: Show state before merge
                            action.log(
                                message_type="debug_before_merge",
                                existing_rows=len(combined_df),
                                existing_columns=combined_df.columns,
                                new_rows=len(df),
                                new_columns=df.columns,
                                merge_key=merge_key
                            )
                            
                            # Convert existing merge key to date or datetime for consistent merging
                            if merge_key in combined_df.columns:
                                try:
                                    use_datetime = (merge_key == "datetime")
                                    if combined_df[merge_key].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                                        if use_datetime:
                                            combined_df = combined_df.with_columns(
                                                pl.from_epoch(pl.col(merge_key), time_unit="ms").alias(merge_key)
                                            )
                                        else:
                                            combined_df = combined_df.with_columns(
                                                pl.from_epoch(pl.col(merge_key), time_unit="ms").dt.date().alias(merge_key)
                                            )
                                    else:
                                        if use_datetime:
                                            combined_df = combined_df.with_columns(
                                                pl.col(merge_key).str.to_datetime()
                                            )
                                        else:
                                            combined_df = combined_df.with_columns(
                                                pl.col(merge_key).str.to_datetime().dt.date()
                                            )
                                except Exception as e:
                                    action.log(message_type="datetime_conversion_error", error=str(e))
                            
                            # Merge with existing data using outer join
                            combined_df = combined_df.join(df, on=merge_key, how="full", coalesce=True)
                            
                            # DEBUG: Show state after merge
                            action.log(
                                message_type="debug_after_merge",
                                csv_name=csv_name,
                                result_rows=len(combined_df),
                                result_columns=combined_df.columns,
                                sample=combined_df.head(3).to_dicts() if len(combined_df) > 0 else []
                            )
                    
                    except Exception as e:
                        action.log(message_type="processing_error", csv_name=csv_name, error=str(e))
                        if required:
                            break
                        else:
                            continue
                
                # Process the combined data if we have any
                if combined_df is not None and not combined_df.is_empty():
                    # DEBUG: Show combined data before processing
                    action.log(
                        message_type="debug_combined_data_before_processing",
                        combo_name=combo_name,
                        rows=len(combined_df),
                        columns=combined_df.columns,
                        sample_head=combined_df.head(5).to_dicts() if len(combined_df) > 0 else []
                    )
                    
                    # Apply output structure configuration
                    output_config = combo_config.get("output_structure", {})
                    
                    # Convert merge key back to datetime for sorting if needed
                    if merge_key and merge_key in combined_df.columns:
                        try:
                            # If it's a date, convert to datetime for proper sorting
                            if combined_df[merge_key].dtype == pl.Date:
                                combined_df = combined_df.with_columns(
                                    pl.col(merge_key).cast(pl.Datetime)
                                )
                        except Exception as e:
                            action.log(message_type="datetime_conversion_for_sort_error", error=str(e))
                    
                    # Sort the data
                    sort_by = output_config.get("primary_sort", merge_key)
                    sort_ascending = output_config.get("sort_ascending", True)
                    if sort_by and sort_by in combined_df.columns:
                        combined_df = combined_df.sort(sort_by, descending=not sort_ascending)
                    
                    # Convert merge key back to string for CSV output
                    # If merge_key is "datetime", preserve full datetime; otherwise convert to date only
                    if merge_key and merge_key in combined_df.columns:
                        try:
                            use_datetime = (merge_key == "datetime")
                            # If it's a datetime, convert to string (preserving time if needed)
                            if combined_df[merge_key].dtype == pl.Datetime:
                                if use_datetime:
                                    # Keep full datetime as string
                                    combined_df = combined_df.with_columns(
                                        pl.col(merge_key).cast(pl.String).alias(merge_key)
                                    )
                                else:
                                    # Convert to date first then to string
                                    combined_df = combined_df.with_columns(
                                        pl.col(merge_key).dt.date().cast(pl.String).alias(merge_key)
                                    )
                            else:
                                combined_df = combined_df.with_columns(
                                    pl.col(merge_key).cast(pl.String)
                                )
                        except Exception as e:
                            action.log(message_type="date_format_error", error=str(e))
                            combined_df = combined_df.with_columns(
                                pl.col(merge_key).cast(pl.String)
                            )
                    
                    # Select final columns if specified
                    final_columns = output_config.get("final_columns", [])
                    if final_columns:
                        available_final_columns = [col for col in final_columns if col in combined_df.columns]
                        if available_final_columns:
                            combined_df = combined_df.select(available_final_columns)
                            action.log(message_type="selected_final_columns", columns=available_final_columns)
                        else:
                            action.log(
                                message_type="no_final_columns_available",
                                requested=final_columns,
                                available=combined_df.columns
                            )
                    
                    # Apply data processing options
                    data_processing = output_config.get("data_processing", {})
                    if data_processing.get("fill_missing_values"):
                        fill_value = data_processing["fill_missing_values"]
                        combined_df = combined_df.fill_null(fill_value)
                    
                    # Interpolate temperature if requested
                    if data_processing.get("interpolate_temperature") and "temperature" in combined_df.columns:
                        action.log(message_type="interpolating_temperature")
                        try:
                            # Replace -1 with null for interpolation
                            combined_df = combined_df.with_columns(
                                pl.when(pl.col("temperature") == -1.0)
                                .then(None)
                                .otherwise(pl.col("temperature"))
                                .alias("temperature")
                            )
                            # Apply linear interpolation
                            combined_df = combined_df.with_columns(
                                pl.col("temperature").interpolate().alias("temperature")
                            )
                            # Round to 2 decimal places
                            combined_df = combined_df.with_columns(
                                pl.col("temperature").round(2).alias("temperature")
                            )
                            action.log(message_type="temperature_interpolated_and_rounded")
                        except Exception as e:
                            action.log(message_type="interpolation_error", error=str(e))
                    
                    # Convert sexual_activity to boolean marker (1 if exists, empty if not)
                    if "sexual_activity" in combined_df.columns:
                        combined_df = combined_df.with_columns(
                            pl.when(pl.col("sexual_activity").is_not_null())
                            .then(pl.lit(1))
                            .otherwise(None)
                            .alias("sexual_activity")
                        )
                    
                    # Calculate sleep duration if requested
                    if data_processing.get("calculate_sleep_duration") and "sleep_start_time" in combined_df.columns and "sleep_end_time" in combined_df.columns:
                        action.log(message_type="calculating_sleep_duration")
                        try:
                            # Parse sleep start and end times as datetime
                            combined_df = combined_df.with_columns([
                                pl.col("sleep_start_time").str.to_datetime().alias("sleep_start_dt"),
                                pl.col("sleep_end_time").str.to_datetime().alias("sleep_end_dt")
                            ])
                            
                            # Calculate duration in seconds
                            combined_df = combined_df.with_columns(
                                (pl.col("sleep_end_dt") - pl.col("sleep_start_dt")).dt.total_seconds().alias("duration_seconds")
                            )
                            
                            # Convert to HH:MM:SS format
                            combined_df = combined_df.with_columns(
                                pl.when(pl.col("duration_seconds").is_not_null())
                                .then(
                                    (pl.col("duration_seconds") // 3600).cast(pl.Int32).cast(pl.Utf8).str.pad_start(2, "0") + ":" +
                                    ((pl.col("duration_seconds") % 3600) // 60).cast(pl.Int32).cast(pl.Utf8).str.pad_start(2, "0") + ":" +
                                    (pl.col("duration_seconds") % 60).cast(pl.Int32).cast(pl.Utf8).str.pad_start(2, "0")
                                )
                                .otherwise(None)
                                .alias("sleep_duration")
                            )
                            
                            # Drop temporary columns
                            combined_df = combined_df.drop(["sleep_start_dt", "sleep_end_dt", "duration_seconds"])
                            
                            # Add sleep_duration to final columns if not already there
                            final_columns = output_config.get("final_columns", [])
                            if final_columns and "sleep_duration" not in final_columns:
                                # Insert sleep_duration after sleep_end_time
                                if "sleep_end_time" in final_columns:
                                    idx = final_columns.index("sleep_end_time") + 1
                                    final_columns.insert(idx, "sleep_duration")
                                else:
                                    final_columns.append("sleep_duration")
                                output_config["final_columns"] = final_columns
                            
                            action.log(message_type="sleep_duration_calculated")
                        except Exception as e:
                            action.log(message_type="sleep_duration_calculation_error", error=str(e))
                    
                    # Calculate exercise duration if requested
                    if data_processing.get("calculate_exercise_duration") and "start_time" in combined_df.columns and "end_time" in combined_df.columns:
                        action.log(message_type="calculating_exercise_duration")
                        try:
                            # Parse start and end times as datetime
                            combined_df = combined_df.with_columns([
                                pl.col("start_time").str.to_datetime().alias("start_dt"),
                                pl.col("end_time").str.to_datetime().alias("end_dt")
                            ])
                            
                            # Calculate duration in seconds
                            combined_df = combined_df.with_columns(
                                (pl.col("end_dt") - pl.col("start_dt")).dt.total_seconds().alias("duration_seconds")
                            )
                            
                            # Convert to HH:MM:SS format
                            combined_df = combined_df.with_columns(
                                pl.when(pl.col("duration_seconds").is_not_null())
                                .then(
                                    (pl.col("duration_seconds") // 3600).cast(pl.Int32).cast(pl.Utf8).str.pad_start(2, "0") + ":" +
                                    ((pl.col("duration_seconds") % 3600) // 60).cast(pl.Int32).cast(pl.Utf8).str.pad_start(2, "0") + ":" +
                                    (pl.col("duration_seconds") % 60).cast(pl.Int32).cast(pl.Utf8).str.pad_start(2, "0")
                                )
                                .otherwise(None)
                                .alias("exercise_duration")
                            )
                            
                            # Drop temporary columns
                            combined_df = combined_df.drop(["start_dt", "end_dt", "duration_seconds"])
                            
                            # Add exercise_duration to final columns if not already there
                            final_columns = output_config.get("final_columns", [])
                            if final_columns and "exercise_duration" not in final_columns:
                                # Insert exercise_duration after end_time
                                if "end_time" in final_columns:
                                    idx = final_columns.index("end_time") + 1
                                    final_columns.insert(idx, "exercise_duration")
                                else:
                                    final_columns.append("exercise_duration")
                                output_config["final_columns"] = final_columns
                            
                            action.log(message_type="exercise_duration_calculated")
                        except Exception as e:
                            action.log(message_type="exercise_duration_calculation_error", error=str(e))
                    
                    # Calculate respiratory duration if requested
                    if data_processing.get("calculate_respiratory_duration") and "respiratory_start_time" in combined_df.columns and "respiratory_end_time" in combined_df.columns:
                        action.log(message_type="calculating_respiratory_duration")
                        try:
                            # Parse respiratory start and end times as datetime
                            combined_df = combined_df.with_columns([
                                pl.col("respiratory_start_time").str.to_datetime().alias("respiratory_start_dt"),
                                pl.col("respiratory_end_time").str.to_datetime().alias("respiratory_end_dt")
                            ])
                            
                            # Calculate duration in seconds
                            combined_df = combined_df.with_columns(
                                (pl.col("respiratory_end_dt") - pl.col("respiratory_start_dt")).dt.total_seconds().alias("duration_seconds")
                            )
                            
                            # Convert to HH:MM:SS format
                            combined_df = combined_df.with_columns(
                                pl.when(pl.col("duration_seconds").is_not_null())
                                .then(
                                    (pl.col("duration_seconds") // 3600).cast(pl.Int32).cast(pl.Utf8).str.pad_start(2, "0") + ":" +
                                    ((pl.col("duration_seconds") % 3600) // 60).cast(pl.Int32).cast(pl.Utf8).str.pad_start(2, "0") + ":" +
                                    (pl.col("duration_seconds") % 60).cast(pl.Int32).cast(pl.Utf8).str.pad_start(2, "0")
                                )
                                .otherwise(None)
                                .alias("respiratory_duration")
                            )
                            
                            # Drop temporary columns
                            combined_df = combined_df.drop(["respiratory_start_dt", "respiratory_end_dt", "duration_seconds"])
                            
                            # Add respiratory_duration to final columns if not already there
                            final_columns = output_config.get("final_columns", [])
                            if final_columns and "respiratory_duration" not in final_columns:
                                # Insert respiratory_duration after respiratory_end_time
                                if "respiratory_end_time" in final_columns:
                                    idx = final_columns.index("respiratory_end_time") + 1
                                    final_columns.insert(idx, "respiratory_duration")
                                else:
                                    final_columns.append("respiratory_duration")
                                output_config["final_columns"] = final_columns
                            
                            action.log(message_type="respiratory_duration_calculated")
                        except Exception as e:
                            action.log(message_type="respiratory_duration_calculation_error", error=str(e))
                    
                    # Remove completely empty rows
                    combined_df = combined_df.filter(~pl.all_horizontal(pl.all().is_null()))
                    
                    # Save to output file
                    output_path = Path(output_file)
                    
                    # DEBUG: Show final data before writing
                    action.log(
                        message_type="debug_final_data_before_write",
                        output_file=str(output_path),
                        rows=len(combined_df),
                        columns=combined_df.columns,
                        sample_final=combined_df.head(5).to_dicts() if len(combined_df) > 0 else []
                    )
                    
                    combined_df.write_csv(output_path)
                    
                    action.log(
                        message_type="output_created",
                        file=output_file,
                        absolute_path=str(output_path.absolute()),
                        rows=len(combined_df),
                        columns=combined_df.columns
                    )
                    
                    # Show sample of the data
                    if len(combined_df) > 0:
                        sample_df = combined_df.head(3)
                        action.log(message_type="sample_data", sample=sample_df.to_dicts())
                else:
                    # DEBUG: Show why there's no data
                    action.log(
                        message_type="debug_no_data_to_output",
                        name=combo_name,
                        combined_df_is_none=combined_df is None,
                        combined_df_is_empty=combined_df.is_empty() if combined_df is not None else "N/A"
                    )
            
            except Exception as e:
                action.log(message_type="combination_processing_error", name=combo_name, error=str(e))


def extract_data(samsung_data_path: Path = Path("./Samsung Health")) -> None:
    """Extract and process Samsung Health data.
    
    Args:
        samsung_data_path: Path to the Samsung Health data directory
    """
    with start_action(action_type="extract_data", path=str(samsung_data_path)) as action:
        # Find Samsung dump directories
        dump_dirs = list(samsung_data_path.glob("*"))
        if not dump_dirs:
            action.log(message_type="error", error="No dump directories found")
            raise ValueError(f"No dump directories found in {samsung_data_path}")
        
        samsung_dump_dir = dump_dirs[0]
        action.log(message_type="found_dumps", count=len(dump_dirs), selected=samsung_dump_dir.name)
        
        # Find all CSV files
        csv_paths = list(samsung_dump_dir.glob("*.csv"))
        action.log(
            message_type="debug_found_csvs", 
            count=len(csv_paths),
            files=[str(p) for p in csv_paths]
        )
        
        # DEBUG: Show cleaned names for all CSV files
        all_cleaned_names = {}
        for csv_path in csv_paths:
            cleaned = clean_csv_name(csv_path.name)
            all_cleaned_names[csv_path.name] = cleaned
        action.log(
            message_type="debug_all_csv_names_cleaned",
            mapping=all_cleaned_names
        )
        
        # Get and display cleaned CSV names in hierarchical structure
        action.log(message_type="filtering_csv_files")
        
        categories = get_cleaned_csv_names(csv_paths)
        if isinstance(categories, dict):
            action.log(message_type="hierarchical_structure_generated")
            
            # Load combination configuration
            combination_config = read_combination_config()
            
            # Process data combinations if configuration exists
            if combination_config:
                action.log(message_type="starting_data_combinations")
                process_data_combinations(combination_config, categories, csv_paths)
            
            # Calculate statistics and print to console
            total_files = 0
            typer.echo("\n" + "="*80)
            typer.echo("📊 AVAILABLE CSV FILES (Hierarchical Structure)")
            typer.echo("="*80 + "\n")
            
            for category, data in categories.items():
                category_total = len(data["subcategories"]) + len(data["files"])
                total_files += category_total
                
                # Print category header
                typer.echo(f"📁 {category}")
                typer.echo(f"   ├─ Subcategories: {len(data['subcategories'])}")
                typer.echo(f"   ├─ Direct files: {len(data['files'])}")
                typer.echo(f"   └─ Total: {category_total}")
                
                # Print subcategories if any
                if data["subcategories"]:
                    typer.echo(f"\n   Subcategories:")
                    for subcat in data["subcategories"]:
                        typer.echo(f"      • {subcat}")
                
                # Print direct files if any
                if data["files"]:
                    typer.echo(f"\n   Files:")
                    for file in data["files"]:
                        typer.echo(f"      • {file}")
                
                typer.echo()  # Empty line between categories
                
                action.log(
                    message_type="category_stats",
                    category=category,
                    subcategories=len(data["subcategories"]),
                    files=len(data["files"]),
                    total=category_total
                )
            
            typer.echo("="*80)
            typer.echo(f"📊 SUMMARY: {len(categories)} categories, {total_files} total files")
            typer.echo("="*80 + "\n")
            
            action.log(message_type="overall_stats", total_categories=len(categories), total_files=total_files)


@app.command("extract")
def main(
    samsung_data_path: str = typer.Argument("./Samsung Health", help="Folder holding the unpacked export folder"),
    log_file: str | None = typer.Option(None, help="Log file (default: logs/samsung_health_extract_<time>.log)"),
) -> None:
    """Combine CSVs as configured in data_combination.json (the original, config-driven extraction).
    
    Args:
        samsung_data_path: Path to the Samsung Health data directory (default: ./Samsung Health)
        log_file: Path to log file (default: auto-generated in logs/ directory)
    """
    data_path = Path(samsung_data_path)
    
    if not data_path.exists():
        typer.echo(f"❌ Error: Directory '{samsung_data_path}' does not exist!", err=True)
        raise typer.Exit(code=1)
    
    if not data_path.is_dir():
        typer.echo(f"❌ Error: '{samsung_data_path}' is not a directory!", err=True)
        raise typer.Exit(code=1)
    
    # Setup logging
    log_path = Path(log_file) if log_file else None
    log_handle = setup_logging(log_path)
    actual_log_file = log_handle.name
    
    typer.echo(f"📝 Logging to: {actual_log_file}")
    
    try:
        extract_data(data_path)
        typer.echo("✅ Data extraction completed successfully!")
        typer.echo(f"📋 Check log file for details: {actual_log_file}")
    finally:
        # Close log file
        log_handle.close()


@app.command("streams")
def streams_command(
    export_path: Path = typer.Argument(..., help="The export folder, or a folder holding exactly one"),
    out_dir: Path = typer.Argument(..., help="Where to write one parquet per stream"),
) -> None:
    """Write each parsed stream (heart rate, step bins, sleep, exercise, period days) to parquet, in local time."""
    from samsung_health import SamsungExport, Streams

    data = Streams.read(SamsungExport.locate(export_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in vars(data).items():
        frame.write_parquet(out_dir / f"{name}.parquet")
        span = ""
        time = next((c for c in ("available", "start", "date") if c in frame.columns), None)
        if time and frame.height:
            span = f"  {frame[time].min()} .. {frame[time].max()}"
        typer.echo(f"{name:20s} {frame.height:>9,d} rows{span}")


@app.command("align")
def align_command(
    export_path: Path = typer.Argument(..., help="The export folder, or a folder holding exactly one"),
    source: Path = typer.Argument(..., help="A parquet or CSV with a naive local timestamp column (a CGM track)"),
    output: Path = typer.Argument(..., help="Where to write it with the wearable columns (.parquet or .csv)"),
    time_column: str = typer.Option("date", help="The timestamp column of SOURCE"),
    overwrite: bool = typer.Option(False, help="Replace wearable columns that already hold data"),
) -> None:
    """Add past-only wearable columns (heart rate, steps, exercise, sleep, menstrual cycle) to a timeline."""
    from samsung_health import SamsungExport, Streams, add_wearables

    frame = pl.read_parquet(source) if source.suffix == ".parquet" else pl.read_csv(source, try_parse_dates=True)
    if frame.schema[time_column] == pl.String:
        frame = frame.with_columns(pl.col(time_column).str.to_datetime(strict=False))
    joined = add_wearables(frame, time_column, Streams.read(SamsungExport.locate(export_path)), overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        joined.write_parquet(output, statistics=True)
    else:
        joined.write_csv(output)
    typer.echo(f"{output}: {joined.height:,d} rows")
    _coverage(joined)


@app.command("grid")
def grid_command(
    export_path: Path = typer.Argument(..., help="The export folder, or a folder holding exactly one"),
    output: Path = typer.Argument(..., help="Where to write the five-minute track (.parquet or .csv)"),
    start: str = typer.Option(..., help="First timestamp, local, e.g. 2024-04-11T00:00"),
    end: str = typer.Option(..., help="Last timestamp, local"),
) -> None:
    """The wearable columns alone on a regular five-minute grid of local time."""
    from samsung_health import SamsungExport, Streams, wearable_columns

    times = pl.datetime_range(datetime.fromisoformat(start), datetime.fromisoformat(end), "5m",
                              time_unit="us", eager=True).alias("date")
    frame = pl.concat([times.to_frame(), wearable_columns(times, Streams.read(SamsungExport.locate(export_path)))],
                      how="horizontal")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output, statistics=True) if output.suffix == ".parquet" else frame.write_csv(output)
    typer.echo(f"{output}: {frame.height:,d} rows")
    _coverage(frame)


def _coverage(frame: pl.DataFrame) -> None:
    from samsung_health import COLUMNS

    for name in COLUMNS:
        filled = frame.height - frame[name].null_count()
        typer.echo(f"  {name:18s} {filled:>9,d} rows filled ({100 * filled / max(frame.height, 1):5.1f} %)")


if __name__ == "__main__":
    app()

