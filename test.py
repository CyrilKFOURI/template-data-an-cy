import ipywidgets as widgets
from IPython.display import display

# Dropdown des marques
brand_dropdown = widgets.Dropdown(
    options=sorted(df["BRAND"].dropna().unique()),
    description="Brand:"
)

# Zone d'affichage
output = widgets.Output()

# Fonction update
def show_models(change):
    output.clear_output()

    selected_brand = change["new"]

    models = (
        df[df["BRAND"] == selected_brand]["MODEL"]
        .dropna()
        .unique()
    )

    with output:
        print(f"Brand: {selected_brand}")
        print(f"Number of models: {len(models)}\n")

        for model in sorted(models):
            print(f"- {model}")

# Trigger quand on change de marque
brand_dropdown.observe(show_models, names="value")

# Initialisation
show_models({"new": brand_dropdown.value})

# Affichage
display(brand_dropdown, output)