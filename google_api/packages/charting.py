# %%
import gradio as gr
import pandas as pd
import plotly.express as px
from plotly.graph_objs._figure import Figure


def chart(
    df: pd.DataFrame,
    x_col: str = "",
    y_col: str = "",
    color_col: str = "",
    agg: str = "",
    chart_type: str = "",
) -> None:
    chart_type_values: list[str] = [
        "line",
        "scatter",
        "bar",
        "area",
        "histogram",
        "box",
        "violin",
        "pie",
        "funnel",
        "sunburst",
    ]
    metrics: list[str] = sorted(df.columns.tolist())
    x_col_values: list[str] = sorted([col for col in metrics if col])
    y_col_values: list[str] = sorted(
        [col for col in metrics]  # if col and df[col].dtype.kind in "if"
    )
    color_col_values: list[str] = sorted(
        [col for col in metrics]  # if col and df[col].dtype.kind not in "if"
    )
    agg_values: list[str] = [
        "sum",
        "mean",
        "median",
        "count",
        "min",
        "max",
        "std",
        "var",
    ]

    def update(
        chart_type: str,
        x_col: str,
        y_col: str,
        color_col: str,
        agg: str,
    ) -> Figure:
        """
        Update the table and plot based on the selected filters.
        """
        group_cols = []
        for col in [x_col, color_col]:
            if col not in group_cols:
                group_cols.append(col)
        df_copy: pd.DataFrame = df[[*set(group_cols + [y_col])]].copy()
        df_copy = df_copy.groupby(group_cols)[y_col].agg(agg).reset_index()

        # Sort by x-axis column first, then by color column if it exists
        sort_cols = [x_col]
        if color_col and color_col != "None" and color_col != x_col:
            sort_cols.append(color_col)

        # Handle numeric strings properly for sorting
        for col in sort_cols:
            try:
                # Try to convert to numeric for proper sorting
                df_copy[col + "_numeric"] = pd.to_numeric(df_copy[col], errors="coerce")
                if df_copy[col + "_numeric"].notna().all():
                    # Replace original column with numeric version for sorting
                    sort_cols[sort_cols.index(col)] = col + "_numeric"
            except:
                pass

        df_copy = df_copy.sort_values(by=sort_cols, ascending=True)

        # Clean up numeric helper columns
        cols_to_drop = [col for col in df_copy.columns if col.endswith("_numeric")]
        df_copy = df_copy.drop(columns=cols_to_drop).reset_index(drop=True)
        kwargs = {
            "data_frame": df_copy,
            "x": x_col,
            "y": y_col,
            "color": color_col if color_col != "None" else None,
            "title": f"{y_col} by {x_col} and {color_col}",
        }
        fig: Figure = getattr(px, chart_type)(**kwargs)

        # Force x-axis to follow the order in the dataframe
        unique_x_values = df_copy[x_col].unique().tolist()
        fig.update_xaxes(categoryorder="array", categoryarray=unique_x_values)

        return fig

    with gr.Blocks() as demo:
        gr.Markdown(value="# Disposable Dashboard with Plotly and Gradio")

        with gr.Row():
            chart_type_dropdown = gr.Dropdown(
                choices=chart_type_values,
                value=chart_type or chart_type_values[0],
                label="Chart Type",
            )
            x_col_dropdown = gr.Dropdown(
                choices=x_col_values,
                value=x_col or x_col_values[0],
                label="X-axis",
            )
            y_col_dropdown = gr.Dropdown(
                choices=y_col_values,
                value=y_col or y_col_values[0],
                label="Y-axis",
            )
            agg_dropdown = gr.Dropdown(
                choices=agg_values,
                value=agg or agg_values[0],
                label="Aggregation",
            )
            color_col_dropdown = gr.Dropdown(
                choices=color_col_values,
                value=color_col or color_col_values[0],
                label="Color by",
            )

        plot_output = gr.Plot(
            value=update(
                chart_type=chart_type_values[0],
                x_col=x_col_values[0],
                y_col=y_col_values[1],
                color_col=color_col_values[2],
                agg=agg_values[0],
            ),
            label="Plot",
        )
        _table_output = gr.Dataframe(
            value=df,
            headers=list(df.columns),
            datatype=["number" if dtype.kind in "if" else "str" for dtype in df.dtypes],
            interactive=True,
        )

        inputs: list[gr.Slider | gr.Dropdown] = [
            chart_type_dropdown,
            x_col_dropdown,
            y_col_dropdown,
            color_col_dropdown,
            agg_dropdown,
        ]
        outputs: list[gr.Dataframe | gr.Plot] = [plot_output]
        for widget in inputs:
            widget.change(fn=update, inputs=inputs, outputs=outputs)
    demo.launch(inbrowser=True)


if __name__ == "__main__":
    df: pd.DataFrame = px.data.gapminder()
    chart(df=df)
