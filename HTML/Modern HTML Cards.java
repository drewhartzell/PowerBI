Orders HTML = 
VAR _CurrentValue = [Orders This Year]
VAR _LastWeekValue = [Orders Previous Year]
VAR _Diff = _CurrentValue - _LastWeekValue
VAR _DiffPct = DIVIDE(_Diff, _LastWeekValue, 0)

-- Abbreviation logic
VAR _Current = 
    SWITCH(
        TRUE(),
        ABS(_CurrentValue) >= 1000000000, FORMAT(_CurrentValue / 1000000000, "0.00") & "B",
        ABS(_CurrentValue) >= 1000000,    FORMAT(_CurrentValue / 1000000, "0.00") & "M",
        FORMAT(_CurrentValue, "#,0")
    )

VAR _LastWeek = 
    SWITCH(
        TRUE(),
        ABS(_LastWeekValue) >= 1000000000, FORMAT(_LastWeekValue / 1000000000, "0.00") & "B",
        ABS(_LastWeekValue) >= 1000000,    FORMAT(_LastWeekValue / 1000000, "0.00") & "M",
        FORMAT(_LastWeekValue, "#,0")
    )

VAR _PctFormatted = FORMAT(ABS(_DiffPct), "0.0%")

-- Arrow color: #B1CCF7 = up (light blue), #112E5A = down (dark navy)
VAR _ArrowColor = 
    SWITCH(TRUE(),
        _Diff > 0, "#B1CCF7",
        _Diff < 0, "#112E5A",
        "gray"
    )

VAR _Icon = 
    SWITCH(TRUE(),
        _Diff > 0, "▲",
        _Diff < 0, "▼",
        "•"
    )

-- Pill background: softer tint matching the arrow direction
VAR _PillBg = 
    SWITCH(TRUE(),
        _Diff > 0, "#1a3a6e",
        _Diff < 0, "#d0e4ff",
        "#eeeeee"
    )

VAR _PillText = 
    SWITCH(TRUE(),
        _Diff > 0, "#B1CCF7",
        _Diff < 0, "#112E5A",
        "#555555"
    )

RETURN
"<div style='padding: 10px 14px 8px 14px; font-family: DIN, sans-serif;'>
    <p style='font-size:14px; margin:0 0 4px 0; color:#888; font-family: DIN, sans-serif;'>
    Number of Orders
    </p>
    <div style='display:flex; align-items:center; gap:10px; flex-wrap:wrap;'>
        <h1 style='margin:0; font-size:28px; font-weight:700; line-height:1; font-family: DIN, sans-serif;'>" & _Current & "</h1>
        <span style='
            display:inline-flex;
            align-items:center;
            gap:4px;
            background:" & _PillBg & ";
            border-radius:4px;
            padding:3px 8px;
            font-size:14px;
            font-weight:600;
            color:" & _PillText & ";
            font-family: DIN, sans-serif;'>
            <span style='color:" & _ArrowColor & "; font-size:11px;'>" & _Icon & "</span>" & _PctFormatted & "
        </span>
    </div>
    <p style='font-size:14px; margin:6px 0 0 0; color:#888; font-family: DIN, sans-serif;'>
        Last Year: " & _LastWeek & "
    </p>
</div>"
