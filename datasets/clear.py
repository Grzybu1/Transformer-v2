with open('datasets\eng_french.csv', encoding="utf8") as original, open("datasets\eng_fr_corrected.csv", 'w', encoding="ascii", errors="ignore") as corrected:
    for line in original:
        if(line.count(',') == 1):
            corrected.write(line)