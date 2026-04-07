// In a card, list the selected value, or X, or Y (in this case, Global or Multiple Regions) //

Selected Region = 
VAR RegionCount = DISTINCTCOUNT(PPV_Combined[Region])
RETURN 
    SWITCH(
        TRUE(),
        RegionCount = 0 || RegionCount = 4, "Global",
        RegionCount = 1, SELECTEDVALUE(PPV_Combined[Region]),
        RegionCount > 1, "Multiple Regions"
    )
