Show Vote Summary = 
SUMMARIZECOLUMNS(
    'title episode'[parentTconst],
    "Total Show Votes",
        SUM ( 'title ratings'[numVotes] )
)
