wrkbk = openpyxl.load_workbook("playstorescrapping.xlsx")
  
sh = wrkbk.active

for i in range(2, sh.max_row+1):    
    for j in range(1, sh.max_column+1):
        cell_obj = sh.cell(row=i, column=j)
        value=cell_obj.value
        cell_obj.value=preprocess(value)
        result=predict(cell_obj.value)
        
        review.append(cell_obj.value)
        sentiment.append(result)

    if result == "positive":
        countpositive=countpositive+1
    else:
        countnegative=countnegative+1