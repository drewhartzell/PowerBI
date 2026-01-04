let
    Source = Csv.Document(File.Contents("C:\Users\andre\Downloads\title.ratings.csv"),[Delimiter=",", Columns=3, Encoding=1252, QuoteStyle=QuoteStyle.None]),
    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{{"tconst", type text}, {"averageRating", type number}, {"numVotes", Int64.Type}}),
    #"Sorted Rows" = Table.Sort(#"Changed Type",{{"numVotes", Order.Descending}})
in
    #"Sorted Rows"
