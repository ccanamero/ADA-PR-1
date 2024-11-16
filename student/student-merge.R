d1=read.table("student-mat.csv",sep=";",header=TRUE)
d2=read.table("student-por.csv",sep=";",header=TRUE)

d3=merge(d1,d2,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
print(nrow(d3)) # 382 students
print(ncol(d3)) # 382 students

# Exportar el DataFrame combinado a un archivo CSV
#write.csv(d3, "merged_students_r.csv", row.names=FALSE)