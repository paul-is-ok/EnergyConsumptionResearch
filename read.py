import xlrd
from pandas import pd

wb = xlrd.open_workbook('newEnergy.xls')
sh = wb.sheet_names()
print(sh)
sh = wb.sheet_by_name('RiceHallElectricMeters')
print(sh.row_values(1))
rows = sh.nrows
for i in range(0, rows):
    print(sh.row_values(i))
