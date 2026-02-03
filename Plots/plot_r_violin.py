"""
R-based split violin plot for LOS data using ggdist package.
"""

import pandas as pd


def los_split_violin_r_plot(data_dict, normalize_by_focus=False):
    """
    Creates a split violin plot with dots using R's gg dist package via rpy2.

    Args:
        data_dict: Dictionary with result data
        normalize_by_focus: If True, divides sum_focus_los by D_focus
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        from rpy2.robjects.conversion import localconverter

        # Import R packages
        base = importr('base')
        ggplot2 = importr('ggplot2')
        dplyr = importr('dplyr')

        try:
            ggdist = importr('ggdist')
        except Exception:
            print("Error: R package 'ggdist' not installed.")
            print("Install with: install.packages('ggdist') in R")
            return None

    except ImportError as e:
        print(f"Error: rpy2 import failed: {e}")
        print("Ensure rpy2 is installed (pip install rpy2) and R is correctly configured (R_HOME).")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during imports: {e}")
        return None

    # Convert dictionary to DataFrame
    df = pd.DataFrame(data_dict)

    # Data preparation
    y_col = "sum_focus_los"
    y_label = "Length of Stay (Days)"

    if normalize_by_focus:
        if 'D_focus' in df.columns:
            d_col = 'D_focus'
        elif 'D_focus_count' in df.columns:
            d_col = 'D_focus_count'
        else:
            d_col = None

        if d_col:
            df[d_col] = pd.to_numeric(df[d_col], errors='coerce')
            df['normalized_los'] = df['sum_focus_los'] / df[d_col]
            y_col = "normalized_los"
            y_label = "Avg Focus Length of Stay (Days per Patient)"

    # Mapping
    df['Service_Model'] = df['OnlyHuman'].map({1: 'Human Only', 0: 'Hybrid'})

    pttr_mapping = {
        'light': 'Light',
        'mp': 'Medium',
        'medium': 'Medium',
        'heavy': 'Heavy'
    }
    df['pttr_clean'] = df['pttr'].map(pttr_mapping).fillna(df['pttr'])

    # Prepare data for R
    plot_data = df[['pttr_clean', 'Service_Model', y_col]].dropna()
    plot_data.columns = ['x', 'group', 'y']

    # Convert to R dataframe using modern context manager approach
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(plot_data)

    # Create R plot
    ro.r.assign('dat', r_df)

    r_code = f'''
    library(ggplot2)
    library(dplyr)
    library(ggdist)

    dat$x <- factor(dat$x, levels = c("Light", "Medium", "Heavy"))
    dat$group <- factor(dat$group, levels = c("Human Only", "Hybrid"))

    med <- dat %>%
      group_by(x, group) %>%
      summarise(y_med = median(y), .groups = "drop") %>%
      mutate(
        x_num = as.numeric(x),
        side_offset = ifelse(group == "Human Only", -0.1, 0.1),
        tick_halfwidth = 0.10,
        x0 = x_num + side_offset - tick_halfwidth,
        x1 = x_num + side_offset + tick_halfwidth
      )

    p <- ggplot(dat, aes(x = x, y = y)) +
      # Use slab for the violin part
      stat_slab(aes(fill = group, side = group), scale = 0.5, alpha = 0.3, color = NA) +
      # Use dots with thin semi-transparent black outlines
      stat_dots(aes(side = group, fill = group), 
                binwidth = NA, dotsize = 0.2, stackratio = 1, alpha = 0.9, overflow = "compress", 
                show.legend = FALSE, color = "#00000080", stroke = 0.1) +
      # Refined median bars: slightly thicker and perfectly horizontal
      geom_segment(data = med, aes(x = x0, xend = x1, y = y_med, yend = y_med),
                   inherit.aes = FALSE, linewidth = 1.5, lineend = "round", color = "black") +
      scale_side_mirrored(start = "topleft") +
      scale_fill_manual(values = c("Human Only" = "#FFC20A", "Hybrid" = "#0C7BDC")) +
      labs(x = NULL, y = "{y_label}", fill = "Service Model") +
      guides(side = "none") +
      theme_minimal(base_size = 10) +
      theme(
        legend.position = "right",
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank()
      )

    ggsave("r_split_violin_plot.png", p, width = 6, height = 3, dpi = 150)
    p
    '''

    try:
        # Check for tikzDevice package in R
        try:
            ro.r('library(tikzDevice)')
            has_tikz = True
        except Exception:
            has_tikz = False
            print("Warning: R package 'tikzDevice' not found. TikZ output will be skipped.")
            print("Install with: install.packages('tikzDevice') in R")

        result = ro.r(r_code)
        print("✓ R plot created: 'r_split_violin_plot.png'")

        if has_tikz:
            # Generate TikZ file
            tikz_export_code = '''
            tikz("r_split_violin_plot.tikz", width = 6, height = 3, standAlone = FALSE)
            print(p)
            dev.off()
            '''
            ro.r(tikz_export_code)
            print("✓ TikZ file created: 'r_split_violin_plot.tikz'")

        # Output LaTeX code for the figure environment
        print("\n" + "=" * 20 + " LaTeX TikZ Code " + "=" * 20)
        print(f"\\begin{{figure}}[htbp]")
        print(f"  \\centering")
        if has_tikz:
            print(f"  \\input{{r_split_violin_plot.tikz}}")
        else:
            print(f"  \\includegraphics[width=\\textwidth]{{r_split_violin_plot.png}}")
        print(f"  \\caption{{{y_label} by Service Model and Category.}}")
        print(f"  \\label{{fig:r_violin_plot}}")
        print(f"\\end{{figure}}")
        print("=" * 60 + "\n")

        return result
    except Exception as e:
        print(f"Error creating R plot: {e}")
        return None


if __name__ == "__main__":
    import glob
    import os

    # Results directory
    results_dir = '../results/cg'
    excel_files = glob.glob(os.path.join(results_dir, '*.xlsx'))
    excel_files = [f for f in excel_files if not os.path.basename(f).startswith('~$')]

    if not excel_files:
        print(f"Error: No Excel files found in '{results_dir}'.")
    else:
        newest_excel = max(excel_files, key=os.path.getmtime)
        print(f"Loading newest file: {newest_excel}")

        try:
            df = pd.read_excel(newest_excel)
            print("Data loaded successfully.")
            data_dict = df.to_dict('list')

            # Create R-based split violin plot
            los_split_violin_r_plot(data_dict, normalize_by_focus=True)

        except Exception as e:
            print(f"Error loading file: {e}")