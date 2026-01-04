let
    Source = Csv.Document(File.Contents("C:\Users\andre\Downloads\title.episode.csv"),[Delimiter=",", Columns=4, Encoding=1252, QuoteStyle=QuoteStyle.None]),
    #"Changed Type" = Table.TransformColumnTypes(Source,{{"Column1", type text}, {"Column2", type text}, {"Column3", type text}, {"Column4", type text}}),
    #"Promoted Headers" = Table.PromoteHeaders(#"Changed Type", [PromoteAllScalars=true]),
    #"Replaced Value" = Table.ReplaceValue(#"Promoted Headers","\N","0",Replacer.ReplaceText,{"seasonNumber", "episodeNumber"}),
    #"Changed Type1" = Table.TransformColumnTypes(#"Replaced Value",{{"tconst", type text}, {"parentTconst", type text}, {"seasonNumber", Int64.Type}, {"episodeNumber", Int64.Type}}),
    #"Sorted Rows" = Table.Sort(#"Changed Type1",{{"seasonNumber", Order.Ascending}})
in
    #"Sorted Rows"
