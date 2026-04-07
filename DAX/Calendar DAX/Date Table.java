// Date Table //

Calendar = 
ADDCOLUMNS (
    CALENDAR (DATE(2022, 1, 1), DATE(2025, 12, 31)),
    "Month Name", FORMAT([Date], "MMMM"),
    "Month Number", MONTH([Date]),
    "Year", YEAR([Date])
