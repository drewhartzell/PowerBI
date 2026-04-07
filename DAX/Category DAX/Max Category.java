// For a certain category, what is the highest average //

Max Rental Category = 
VAR CategoryTable =
    ADDCOLUMNS(
        VALUES('Data'[Rental Category]),
        "@Rate", CALCULATE(AVERAGE('Data'[Daily Rate Average]))
    )
VAR MaxRate =
    MAXX(CategoryTable, [@Rate])
RETURN
MAXX(
    FILTER(CategoryTable, [@Rate] = MaxRate),
    'Data'[Rental Category]
)
