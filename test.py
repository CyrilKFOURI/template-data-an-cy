import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML

def compare_brand_models(
    df1,
    df2,
    brand_col_df1,
    model_col_df1,
    brand_col_df2,
    model_col_df2,
    name1="Dataset 1",
    name2="Dataset 2"
):

    # -----------------------------
    # Brands
    # -----------------------------
    brands1 = set(df1[brand_col_df1].dropna().astype(str).unique())
    brands2 = set(df2[brand_col_df2].dropna().astype(str).unique())

    common_brands = brands1 & brands2
    only_1 = brands1 - brands2
    only_2 = brands2 - brands1

    print("========== BRAND COMPARISON ==========\n")

    if brands1 == brands2:
        print(f"✅ Exact same brands ({len(brands1)})\n")
    else:
        print("❌ Differences detected\n")

        print(f"Brands only in {name1}:")
        print(sorted(only_1) if only_1 else "None")

        print(f"\nBrands only in {name2}:")
        print(sorted(only_2) if only_2 else "None")

        print(f"\nCommon brands: {len(common_brands)}\n")

    # -----------------------------
    # Dropdown
    # -----------------------------
    dropdown = widgets.Dropdown(
        options=sorted(common_brands),
        description="Brand:",
        layout=widgets.Layout(width="400px")
    )

    output = widgets.Output()

    # -----------------------------
    # Update function
    # -----------------------------
    def update(change):

        output.clear_output()

        brand = change["new"]

        models1 = set(
            df1[df1[brand_col_df1] == brand][model_col_df1]
            .dropna()
            .astype(str)
            .unique()
        )

        models2 = set(
            df2[df2[brand_col_df2] == brand][model_col_df2]
            .dropna()
            .astype(str)
            .unique()
        )

        common_models = models1 & models2

        # LEFT
        html_left = ""

        for model in sorted(models1):

            color = "green" if model in common_models else "red"

            html_left += f"""
            <li style="color:{color}">
                {model}
            </li>
            """

        # RIGHT
        html_right = ""

        for model in sorted(models2):

            color = "green" if model in common_models else "red"

            html_right += f"""
            <li style="color:{color}">
                {model}
            </li>
            """

        html = f"""
        <h2>{brand}</h2>

        <table style="width:100%; border-collapse:collapse;" border="1">

            <tr>
                <th>{name1}</th>
                <th>{name2}</th>
            </tr>

            <tr>

                <td valign="top" style="padding:10px;">
                    <b>Models: {len(models1)}</b>
                    <ul>
                        {html_left}
                    </ul>
                </td>

                <td valign="top" style="padding:10px;">
                    <b>Models: {len(models2)}</b>
                    <ul>
                        {html_right}
                    </ul>
                </td>

            </tr>

        </table>
        """

        with output:
            display(HTML(html))

    dropdown.observe(update, names="value")

    update({"new": dropdown.value})

    display(dropdown, output)




compare_brand_models(
    df1=df_a,
    df2=df_b,

    brand_col_df1="BRAND",
    model_col_df1="MODEL",

    brand_col_df2="MAKE",
    model_col_df2="CAR_MODEL",

    name1="Source A",
    name2="Source B"
)