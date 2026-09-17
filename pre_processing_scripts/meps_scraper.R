# Load the haven package
library(haven)

# Load the SSP file
file_2014 <- "data/MEPS_data_2014.ssp"
file_2016 <- "data/MEPS_data_2016.ssp"

pop_file_2014 = "data/MEPS_pop_data_2014.ssp"
pop_file_2016 = "data/MEPS_pop_data_2016.ssp"

# Read the SSP file
data_2014 <- read_xpt(file_2014)
data_2016 <- read_xpt(file_2016)

pop_data_2014 = read_xpt(pop_file_2014)
pop_data_2016 = read_xpt(pop_file_2016)

# write files to txt 
write.table(data_2014, file = "data/MEPS_data_2014.txt", sep = "\t",
            row.names = TRUE, col.names = NA)
write.table(data_2016, file = "data/MEPS_data_2016.txt", sep = "\t",
            row.names = TRUE, col.names = NA)

write.table(pop_data_2014, file = "data/MEPS_pop_data_2014.txt", sep = "\t",
            row.names = TRUE, col.names = NA)
write.table(pop_data_2016, file = "data/MEPS_pop_data_2016.txt", sep = "\t",
            row.names = TRUE, col.names = NA)