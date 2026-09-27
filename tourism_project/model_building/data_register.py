
import pandas as pd

# Path to the dataset in the repository
DATA_PATH = "tourism_project/data/tourism.csv"

# Expected columns based on the project data dictionary
EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch"
]

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Validate the expected columns
missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]

if missing_columns:
    print("Missing columns:", missing_columns)
else:
    print("All expected columns are present.")

# Print a short dataset summary
print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
