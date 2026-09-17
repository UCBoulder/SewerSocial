# Functions to use in meps_processor_yyyy.R scripts

################################################################################
# remove_drugs() function:
#   * Description: Takes in a large list of pharmaceuticals and a smaller list
#                from within the larger list to remove. Smaller list is removed
#                from the the larger list.
#   * Input: 
#       * full_drug_list: class = list, larger list that needs truncating
#       * removal_drug_list: class = list, shorter list to be truncated from
#                            larger list
#
#   * Output:
#       * full_drug_list: class = list, truncated larger list
################################################################################
remove_drugs = function(full_drug_list, removal_drug_list){
  for (i in 1:length(removal_drug_list)){
    full_drug_list = full_drug_list [ !full_drug_list == removal_drug_list[i]]
  }
  return(full_drug_list)
}

################################################################################
# fix_drug_coding() function:
#   * Description: Corrects entries that have been miscoded in the MEPS data.
#                  Note: some entries required special treatment outside this
#                        function.
#   * Input: 
#       * df: class = dataframe, dataframe with incorrect entries
#       * column: class = vector, column of dataframe with miscoded entries
#       * colname: class = character, name of column with miscoded entries
#       * incorrect: class = many possible, incorrect entry contents
#       * correct: class = many possible, correct entry contents
#
#   * Output:
#       * df: class = dataframe, dataframe with correct entries replacing 
#             incorrect ones
################################################################################
fix_drug_coding = function(df, column, colname, incorrect, correct){
  df = df %>%
    mutate(column = case_when(column==incorrect ~ correct,
                              .default=column)) %>%
    relocate(column, .before=colname) %>%
    select(-c(colname))
  names(df)[names(df)=="column"] = colname
  return(df)
}

################################################################################
# remove_entries() function:
#   * Description: Removes entries from dataframe when they are incorrect.
#   * Input: 
#       * df: class = dataframe, dataframe with incorrect entries
#       * column: class = vector, column of dataframe with incorrect entries
#       * incorrect: class = many possible, incorrect entry contents
#
#   * Output:
#       * df: class = dataframe, dataframe with incorrect entries removed
################################################################################
remove_entries = function(df, column, incorrect){
  df = subset(df, column != incorrect)
  return(df)
}