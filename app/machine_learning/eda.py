import json
import os
from datetime import datetime
from zipfile import ZipFile

import pandas as pd
from ydata_profiling import ProfileReport

from app.config.paths import *


def generate_eda(file_path: str, original_filename: str):
    df = pd.read_csv(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"eda_{os.path.splitext(original_filename)[0]}_{timestamp}"

    html_file = os.path.join(OUTPUTS_EDA, f"{base_name}.html")
    json_file = os.path.join(OUTPUTS_EDA, f"{base_name}.json")

    profile = ProfileReport(df, title="EDA Report", minimal=True)
    profile.to_file(html_file)

    summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict()
    }
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    zip_file = os.path.join(OUTPUTS_ZIPS, f"{base_name}.zip")
    with ZipFile(zip_file, "w") as zipf:
        zipf.write(html_file, os.path.basename(html_file))
        zipf.write(json_file, os.path.basename(json_file))

    return zip_file, summary
