import streamlit as st
import pandas as pd

st.set_page_config(page_title="Marketplace Listing Checker", layout="wide")

st.title("📦 Marketplace Listing Checker")

# -------------------------------
# FILE UPLOAD
# -------------------------------
st.sidebar.header("Upload Files")

catalog_file = st.sidebar.file_uploader("Upload Catalog File", type=["xlsx", "csv"])
marketplace_file = st.sidebar.file_uploader("Upload Marketplace File", type=["xlsx", "csv"])

marketplace_name = st.sidebar.selectbox(
    "Select Marketplace",
    ["Lazada", "Shopee", "Zalora", "Tokopedia", "Generic"]
)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def norm_ean(x):
    try:
        return str(int(float(x))).strip()
    except:
        return str(x).strip()

def aggregate_marketplace(df, sku_col, status_col, name_col):
    return df.groupby(sku_col).agg({
        status_col: lambda x: "active" if "active" in list(x) else "inactive",
        name_col: "first"
    }).reset_index()

# -------------------------------
# MAIN LOGIC
# -------------------------------
if catalog_file and marketplace_file:

    df_catalog = load_file(catalog_file)
    df_market = load_file(marketplace_file)

    st.subheader("🔍 Column Selection")

    col1, col2 = st.columns(2)

    with col1:
        catalog_sku_col = st.selectbox("Catalog SKU/EAN Column", df_catalog.columns)

    with col2:
        market_sku_col = st.selectbox("Marketplace SKU Column", df_market.columns)

    status_col = st.selectbox(
        "Marketplace Status Column (optional)",
        ["None"] + list(df_market.columns)
    )

    name_col_catalog = st.selectbox(
        "Catalog Product Name Column (optional)",
        ["None"] + list(df_catalog.columns)
    )

    name_col_market = st.selectbox(
        "Marketplace Product Name Column (optional)",
        ["None"] + list(df_market.columns)
    )

    # -------------------------------
    # PROCESS
    # -------------------------------
    if st.button("🚀 Run Analysis"):

        # Normalize
        df_catalog["SKU"] = df_catalog[catalog_sku_col].apply(norm_ean)
        df_market["SKU"] = df_market[market_sku_col].apply(norm_ean)

        if status_col != "None":
            df_market["status"] = df_market[status_col].astype(str).str.lower().str.strip()
        else:
            df_market["status"] = "active"

        if name_col_market != "None":
            df_market["product_name"] = df_market[name_col_market]
        else:
            df_market["product_name"] = ""

        # -------------------------------
        # AGGREGATE (CRITICAL FIX)
        # -------------------------------
        df_market_agg = aggregate_marketplace(
            df_market,
            "SKU",
            "status",
            "product_name"
        )

        market_lookup = df_market_agg.set_index("SKU").to_dict("index")

        # -------------------------------
        # BUILD RESULT
        # -------------------------------
        results = []

        for _, row in df_catalog.iterrows():
            sku = row["SKU"]

            product_name = ""
            if name_col_catalog != "None":
                product_name = row[name_col_catalog]

            market_data = market_lookup.get(sku)

            if market_data:
                if market_data["status"] == "active":
                    status = "Listed"
                else:
                    status = "Inactive"
            else:
                status = "Not Listed"

            results.append({
                "SKU": sku,
                "Product Name": product_name,
                "Marketplace Status": status
            })

        df_final = pd.DataFrame(results)

        # -------------------------------
        # METRICS
        # -------------------------------
        total = len(df_final)
        listed = (df_final["Marketplace Status"] == "Listed").sum()
        not_listed = (df_final["Marketplace Status"] == "Not Listed").sum()
        inactive = (df_final["Marketplace Status"] == "Inactive").sum()

        # -------------------------------
        # DISPLAY SUMMARY
        # -------------------------------
        st.subheader("📊 Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total SKUs", total)
        col2.metric("✅ Listed", listed)
        col3.metric("❌ Not Listed", not_listed)
        col4.metric("⚠️ Inactive", inactive)

        # -------------------------------
        # TABLES
        # -------------------------------
        st.subheader("❌ Not Listed")
        st.dataframe(df_final[df_final["Marketplace Status"] == "Not Listed"])

        st.subheader("⚠️ Inactive")
        st.dataframe(df_final[df_final["Marketplace Status"] == "Inactive"])

        st.subheader("✅ Listed")
        st.dataframe(df_final[df_final["Marketplace Status"] == "Listed"])

        # -------------------------------
        # DOWNLOAD
        # -------------------------------
        st.download_button(
            "⬇️ Download Full Report",
            df_final.to_csv(index=False),
            file_name="listing_analysis.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Upload both Catalog and Marketplace files to begin")
