import ipywidgets as widgets
from IPython.display import display

def brand_model_widget(df, brand_col, model_col):

    # Dropdown des brands
    brand_dropdown = widgets.Dropdown(
        options=sorted(df[brand_col].dropna().unique()),
        description="Brand:",
        layout=widgets.Layout(width="300px")
    )

    output = widgets.Output()

    # Fonction d'affichage
    def show_models(change):

        output.clear_output()

        selected_brand = change["new"]

        models = (
            df[df[brand_col] == selected_brand][model_col]
            .dropna()
            .astype(str)
            .unique()
        )

        with output:
            print(f"Brand: {selected_brand}")
            print(f"Number of models: {len(models)}\n")

            for model in sorted(models):
                print(f"- {model}")

    # event listener
    brand_dropdown.observe(show_models, names="value")

    # init display
    show_models({"new": brand_dropdown.value})

    display(brand_dropdown, output)


brand_model_widget(df, "BRAND", "MODEL")