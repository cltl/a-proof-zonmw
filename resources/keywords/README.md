Keywords
========
The files contain keywords for each of the 9 domains; these keywords are used to select notes for annotation (see [data_process_to_inception](../../data_process/data_process_to_inception)).

Each keyword has a **regex_template_id**:

- Keywords with "0" are used as stems, i.e. they will match any word that starts with the given string. For example the keyword "vermoei" will match with "vermoeid", " vermoeidheid ", "vermoeidheidsklachten" etc.
- Keywords with "1" will match only the exact word as it appears. For example, "blij" will not match "blijven".
- Keywords with "2" are two-word expressions that will match any sentence where these two words appear in any order, optionally with other words in between them. For example, the keyword "valt af" will match "valt niet erg af".
