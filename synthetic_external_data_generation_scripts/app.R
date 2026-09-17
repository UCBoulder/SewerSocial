################################################################################
# Title: app.R

# Description:
# Generates the web-based SeweRx R Shiny app. Provides user instructions and 
# allows them to input their own facility data including daily flow or design
# capacity and sewershed population size. Runs PharmFlush model to predict the
# mass loads for the specific water resource recovery facility. Allows the user
# to download PharmUse database and PharmFlush output as csv files. Provides 
# useful data visualizations and allows the user to generate synthetic datasets
# representing their sewershed population demographics for use in the PharmShed
# model (under review for publication and to be converted into a Python package).

################################################################################

# import libraries
library(shiny)
library(plotly)
library(bslib)
library(DT)
library(tidyverse)
library(RColorBrewer)
library(shinycssloaders) 
library(readxl)

# load necessary functions
source("pharmflush_function.R")

################################################################################
# user interface
################################################################################
ui <- fluidPage(
  theme = bs_theme(version = 5, bootswatch = "minty"),
  
  titlePanel("SeweRx"),
  
  sidebarLayout(
    
    # ---------------- SIDEBAR ----------------
    sidebarPanel(
      width = 3,
      
      accordion(
        id = "help_section",
        
        accordion_panel("What is the SeweRx app?",
                        p("SeweRx consists of three modules, PharmUse, PharmFlush, and PharmShed, and continues to grow every day."),
                        p("PharmUse is a database containing data on pharmaceutical consumption, excretion, fate and transport properties, and toxicity for 313 commonly-prescribed pharmaceuticals."),
                        p("PharmFlush is a predictive modeling tool trained on PharmUse data and built into SeweRx. PharmFlush estimates national baseline pharmaceutical mass loads in wastewater influent based on user-defined flow and population parameters."),
                        p("PharmShed is a new prediction tool that is under construction as a Python package. The Shiny app includes a user-friendly platform to generate a synthetic dataset representative of your sewershed population's demographics."),
                        p("To use PharmFlush on SeweRx, enter your WRRF's flow capacity and sewershed size, then click 'Run Model'. This will generate the predicted national baseline pharmaceutical loads at the scale of your WRRF's flows and population."),
                        p("To generate the synthetic population needed to run PharmShed, enter the sewershed size on the app's main page and then go to the SewerShed Population tab. Enter percentages for each demographic category. Click 'Generate Synthetic Data'. Download the resulting CSV file. Further instructions to follow on integration with the PharmShed package."),
                        p("SeweRx assists with treatment priortization and selection of indicator compounds for process monitoring. You can do this by comparing your WRRF's pharmaceutical mass loads against the national average predicted by PharmFlush. We hope this tool will be helpful as more water reuse regulations are passed."),
                        p("For assistance, contact Vanessa Maybruck at vanessa.maybruck@colorado.edu.")
        ),
        accordion_panel("Inputs Explained",
                        tags$ul(
                          tags$li(strong("Daily Flow or Flow Capacity of WRRF (L/day):"), "Typical flow associated with your WRRF or design capacity of your WRRF. Use typical flows for the best results. The default value (34,225,766 L/d) is the flow capacity calculated from the average daily wastewater volume produced per individual in the U.S. (310.404 L/capita/day) and the average sewershed population from the validation set used for PharmFlush (110,262 people)."),
                          tags$li(strong("Sewershed Size:"), "Total number of people contributing to flows in your WRRF. The default value is the average sewershed size in the validation set used in PharmFlush (110,262 people).")
                        )
        )
      ),
      
      hr(),
      
      h5("Core Inputs"),
      
      fileInput(
        "file",
        "Upload CSV / Excel File",
        accept = c(".csv", ".xlsx", ".xls")
      ),
      
      tableOutput("preview"),
      
      hr(),
      
      numericInput("flow_capacity", "Flow Capacity (L/day)", value = 34225766),
      numericInput("population_served", "Sewershed Size", value = 110262),
      
      actionButton("run_model", "Run Model", 
                   class = "btn-primary", width = "100%"),
      
      br(), br(),
      
      downloadButton("download_full_data", 
                     "Download Results", 
                     class = "btn btn-success w-100"),
      
      hr(),
      
      downloadButton("download_file", 
                     "Download PharmUse Data",
                     class = "btn btn-info w-100")
    ),
    
    
    # ---------------- MAIN PANEL ----------------
    mainPanel(
      
      tabsetPanel(
        
        # ---------- RESULTS ----------
        tabPanel("Model Results",
                 
                 br(),
                 
                 card(
                   card_header("National Baseline Mass Loads"),
                   card_body(
                     withSpinner(
                       DT::dataTableOutput("average_profile"),
                       type = 6, color = "#78c2ad"
                     )
                   )
                 ),
                 
                 br(),
                 
                 card(
                   card_header("Reported vs Predicted Heatmap"),
                   card_body(
                     withSpinner(
                       plotOutput("heatmap_plot", height = "500px"),
                       type = 6, color = "#78c2ad"
                     ),
                     br(),
                     tableOutput("underpredicted_table")
                   )
                 ),
                 
                 br(),
                 
                 card(
                   card_header("Scatterplots"),
                   card_body(
                     withSpinner(
                       plotlyOutput("scatter_plot", height = "500px"),
                       type = 6, color = "#78c2ad"
                     ),
                   )
                 ),
                 br(),
                 
                 card(
                   card_header("Pharmaceutical Compatibility Check"),
                   card_body(
                     
                     withSpinner(
                       DT::dataTableOutput("pharma_support_table"),
                       type = 6, color = "#78c2ad"
                     ),
                     
                     br(),
                     
                     fluidRow(
                       column(6,
                              tags$div(
                                style = "background-color:#e6f4ea; padding:12px; border-radius:10px; text-align:center;",
                                tags$b("Supported"),
                                br(),
                                textOutput("supported_count")
                              )
                       ),
                       column(6,
                              tags$div(
                                style = "background-color:#fdecea; padding:12px; border-radius:10px; text-align:center;",
                                tags$b("Not Supported"),
                                br(),
                                textOutput("unsupported_count")
                              )
                       )
                     )
                   )
                 )
        ),
        
        
        # ---------- SYNTHETIC POP ----------
        tabPanel("Synthetic Population",
                 
                 br(),
                 
                 card(
                   card_header("Demographics"),
                   card_body(
                     
                     fluidRow(
                       column(6,
                              h5("Gender (%)"),
                              numericInput("pct_male", "Male", 50, 0, 100),
                              numericInput("pct_female", "Female", 50, 0, 100)
                       ),
                       
                       column(6,
                              h5("Insurance (%)"),
                              numericInput("pct_uninsured", "Uninsured", 10),
                              numericInput("pct_public", "Public", 30),
                              numericInput("pct_private", "Private", 60)
                       )
                     ),
                     
                     hr(),
                     
                     h5("Race / Ethnicity (%)"),
                     fluidRow(
                       column(4, numericInput("pct_white", "Non-Hispanic White", 40)),
                       column(4, numericInput("pct_hispanic", "Hispanic", 20)),
                       column(4, numericInput("pct_black", "Non-Hispanic Black", 15)),
                       column(4, numericInput("pct_asian", "Non-Hispanic Asian", 15)),
                       column(4, numericInput("pct_other", "Non-Hispanic Other", 10))
                     )
                   )
                 ),
                 
                 br(),
                 
                 # Modified to have 9 slots for Age
                 card(
                   card_header("Age Distribution"),
                   card_body(
                     fluidRow(
                       column(4, textInput("age1_range", "Range 1", "0-18"), numericInput("pct_age1", "%", 15)),
                       column(4, textInput("age2_range", "Range 2", "19-30"), numericInput("pct_age2", "%", 15)),
                       column(4, textInput("age3_range", "Range 3", "31-40"), numericInput("pct_age3", "%", 10)),
                       column(4, textInput("age4_range", "Range 4", "41-50"), numericInput("pct_age4", "%", 10)),
                       column(4, textInput("age5_range", "Range 5", "51-60"), numericInput("pct_age5", "%", 10)),
                       column(4, textInput("age6_range", "Range 6", "61-70"), numericInput("pct_age6", "%", 10)),
                       column(4, textInput("age7_range", "Range 7", "71-80"), numericInput("pct_age7", "%", 10)),
                       column(4, textInput("age8_range", "Range 8", "81-90"), numericInput("pct_age8", "%", 10)),
                       column(4, textInput("age9_range", "Range 9", "91-100"), numericInput("pct_age9", "%", 10))
                     )
                   )
                 ),
                 
                 br(),
                 
                 # Modified to have 16 slots for Income
                 card(
                   card_header("Income Distribution"),
                   card_body(
                     fluidRow(
                       column(3, textInput("inc1_range", "Range 1", "0-10000"), numericInput("pct_income1", "%", 10)),
                       column(3, textInput("inc2_range", "Range 2", "10001-20000"), numericInput("pct_income2", "%", 10)),
                       column(3, textInput("inc3_range", "Range 3", "20001-30000"), numericInput("pct_income3", "%", 10)),
                       column(3, textInput("inc4_range", "Range 4", "30001-40000"), numericInput("pct_income4", "%", 10)),
                       column(3, textInput("inc5_range", "Range 5", "40001-50000"), numericInput("pct_income5", "%", 10)),
                       column(3, textInput("inc6_range", "Range 6", "50001-60000"), numericInput("pct_income6", "%", 5)),
                       column(3, textInput("inc7_range", "Range 7", "60001-70000"), numericInput("pct_income7", "%", 5)),
                       column(3, textInput("inc8_range", "Range 8", "70001-80000"), numericInput("pct_income8", "%", 5)),
                       column(3, textInput("inc9_range", "Range 9", "80001-90000"), numericInput("pct_income9", "%", 5)),
                       column(3, textInput("inc10_range", "Range 10", "90001-100000"), numericInput("pct_income10", "%", 5)),
                       column(3, textInput("inc11_range", "Range 11", "100001-120000"), numericInput("pct_income11", "%", 5)),
                       column(3, textInput("inc12_range", "Range 12", "120001-140000"), numericInput("pct_income12", "%", 5)),
                       column(3, textInput("inc13_range", "Range 13", "140001-160000"), numericInput("pct_income13", "%", 5)),
                       column(3, textInput("inc14_range", "Range 14", "160001-180000"), numericInput("pct_income14", "%", 5)),
                       column(3, textInput("inc15_range", "Range 15", "180001-200000"), numericInput("pct_income15", "%", 5)),
                       column(3, textInput("inc16_range", "Range 16", "200001-250000"), numericInput("pct_income16", "%", 5))
                     )
                   )
                 ),
                 
                 br(),
                 
                 actionButton("generate_synth", 
                              "Generate Synthetic Dataset",
                              class = "btn-success w-100"),
                 
                 br(), br(),
                 
                 downloadButton("download_synthetic_csv", "Download Synthetic CSV", class = "btn btn-outline-success w-100"),
                 
                 br(), br(),
                 
                 tableOutput("synthetic_table")
        )
      )
    )
  )
)

################################################################################
# server
################################################################################
server <- function(input, output, session) {
  
  model_results_reactive <- reactiveVal(NULL)
  
  # 1. Trigger the model calculation
  observeEvent(input$run_model, {
    id <- showNotification("Calculating...", duration = NULL, closeButton = FALSE)
    on.exit(removeNotification(id), add = TRUE)
    
    results <- pharmflush_model(
      flow_capacity = input$flow_capacity,
      population_served = input$population_served
    )
    
    model_results_reactive(results)
  })
  
  processed_data <- reactive({
    res <- model_results_reactive()
    req(res, res$average_profile)
    
    df <- as.data.frame(res$average_profile)
    
    df %>%
      dplyr::rename(
        `Predicted Mass Load (ug/capita/day)` = Average_Predicted_Mass_Load,
        Pharmaceutical = Drug
      ) %>%
      mutate(`Predicted Mass Load (ug/capita/day)` = as.numeric(`Predicted Mass Load (ug/capita/day)`)) %>%
      select(Pharmaceutical, `Predicted Mass Load (ug/capita/day)`)
  })
  
  
  uploaded_data <- reactive({
    req(input$file)
    
    ext <- tools::file_ext(input$file$name)
    
    df <- if (ext == "csv") {
      read.csv(input$file$datapath, stringsAsFactors = FALSE)
    } else {
      readxl::read_excel(input$file$datapath)
    }
    
    # must have these columns
    validate(
      need(all(c("Pharmaceutical", "Reported_mass_load") %in% names(df)),
           "File must have columns: Pharmaceutical, Reported_mass_load")
    )
    
    # convert ND → NA, numbers stay numbers
    df$Reported_mass_load[df$Reported_mass_load == "ND"] <- NA
    df$Reported_mass_load <- as.numeric(df$Reported_mass_load)
    
    df
  })
  
  
  
  #supported pharmaceuticals
  pharma_support_table <- reactive({
    req(uploaded_data(), processed_data())
    
    supported <- processed_data()$Pharmaceutical
    user <- unique(uploaded_data()$Pharmaceutical)
    
    data.frame(
      Pharmaceutical = user,
      Status = ifelse(user %in% supported, "Supported", "Not Supported")
    )
  })
  
  output$pharma_support_table <- DT::renderDataTable({
    df <- pharma_support_table()
    
    DT::datatable(df, options = list(pageLength = 10)) %>%
      DT::formatStyle(
        "Status",
        target = "row",
        backgroundColor = DT::styleEqual(
          c("Supported", "Not Supported"),
          c("#d4edda", "#f8d7da")
        )
      )
  })
  
  output$supported_count <- renderText({
    df <- pharma_support_table()
    req(df)
    
    paste(sum(df$Status == "Supported"), "used by model")
  })
  
  output$unsupported_count <- renderText({
    df <- pharma_support_table()
    req(df)
    
    paste(sum(df$Status == "Not Supported"), "ignored")
  })
  
  
  #creating comparison data
  comparison_data <- reactive({
    req(processed_data())
    
    if (is.null(input$file)) {
      return(processed_data())
    }
    
    processed_data() %>%
      left_join(uploaded_data(), by = "Pharmaceutical") %>%
      mutate(
        Difference_Reported_minus_Predicted =
          Reported_mass_load - `Predicted Mass Load (ug/capita/day)`,
        Ratio_Reported_to_Predicted =
          Reported_mass_load / `Predicted Mass Load (ug/capita/day)`
      )
  })
  
  output$preview <- renderTable({
    head(uploaded_data())
  })
  
  output$average_profile <- DT::renderDataTable({
    df <- comparison_data()
    
    DT::datatable(
      df,
      rownames = FALSE,
      options = list(
        pageLength = 10,
        autoWidth = TRUE,
        scrollX = TRUE
      )
    ) %>%
      DT::formatRound(
        columns = intersect(
          c(
            "Predicted Mass Load (ug/capita/day)",
            "Reported_mass_load",
            "Difference_Reported_minus_Predicted",
            "Ratio_Reported_to_Predicted"
          ),
          colnames(df)
        ),
        digits = 2
      )
  })
  
  output$download_full_data <- downloadHandler(
    filename = function() { paste0("pharmflush_output_", Sys.Date(), ".csv") },
    content = function(file) {
      write.csv(processed_data(), file, row.names = FALSE)
    }
  )
  
  output$download_file <- downloadHandler(
    filename = function() { "pharmuse.csv" },
    content = function(file) { file.copy("pharmuse.csv", file) }
  )
  
  output$heatmap_plot <- renderPlot({
    
    req(processed_data(), uploaded_data())
    
    # - Join predicted + reported -
    heatmap_df <- processed_data() %>%
      left_join(uploaded_data(), by = "Pharmaceutical") %>%
      mutate(
        Cell_Category = case_when(
          is.na(Reported_mass_load) ~ "Not Detected / Missing",
          Reported_mass_load == 0 ~ "Not Detected / Missing",
          TRUE ~ "Measured"
        ),
        Ratio = ifelse(
          Cell_Category == "Measured",
          `Predicted Mass Load (ug/capita/day)` / Reported_mass_load,
          NA_real_
        ),
        Log10_Ratio = ifelse(
          !is.na(Ratio) & Ratio > 0,
          log10(Ratio),
          NA_real_
        ),
        Abs_Log10_Ratio = abs(Log10_Ratio),
        Major_Discrepancy = !is.na(Log10_Ratio) & Log10_Ratio <= -1
      )
    
    
    
    top_n <- 100  # adjust as needed
    
    heatmap_df <- heatmap_df %>%
      filter(!is.na(Reported_mass_load)) %>%        # only measured
      arrange(desc(Abs_Log10_Ratio)) %>%             # most discrepant first
      slice_head(n = top_n) %>%
      mutate(
        Pharmaceutical = factor(
          Pharmaceutical,
          levels = rev(unique(Pharmaceutical))
        )
      )
    
    # ---- Split categories ----
    data_measured <- heatmap_df %>% filter(Cell_Category == "Measured")
    data_missing <- heatmap_df %>% filter(Cell_Category != "Measured")
    
    # ---- Heatmap ----
    ggplot(heatmap_df, aes(
      x = "Your Facility",
      y = Pharmaceutical
    )) +
      geom_tile(
        data = data_missing,
        fill = "gray85",
        color = "white",
        linewidth = 0.2
      ) +
      geom_tile(
        data = data_measured,
        aes(fill = Abs_Log10_Ratio),
        color = "white",
        linewidth = 0.2
      ) +
      geom_point(
        data = data_measured %>% filter(Major_Discrepancy),
        shape = 17,
        color = "red",
        size = 2
      ) +
      scale_fill_gradient(
        low = "lightblue",
        high = "darkblue",
        name = "|log10(Predicted / Reported)|"
      ) +
      labs(
        x = "",
        y = "Pharmaceutical",
        caption = "Red triangles indicate Reported > 10× Predicted"
      ) +
      theme_minimal(base_size = 20) +
      theme(
        axis.text.y = element_text(size = 20),
        axis.text.x = element_blank(),
        panel.grid = element_blank(),
        legend.position = "right"
      )
    
  })
  
  # under -1 table
  output$underpredicted_table <- renderTable({
    
    req(processed_data(), uploaded_data())
    
    processed_data() %>%
      left_join(uploaded_data(), by = "Pharmaceutical") %>%
      mutate(
        Ratio = `Predicted Mass Load (ug/capita/day)` / Reported_mass_load,
        Log10_Ratio = log10(Ratio)
      ) %>%
      filter(
        !is.na(Log10_Ratio),
        Log10_Ratio < -1
      ) %>%
      arrange(Log10_Ratio) %>%
      select(
        Pharmaceutical,
        Predicted = `Predicted Mass Load (ug/capita/day)`,
        Reported = Reported_mass_load,
        Log10_Ratio
      )
    
  }, digits = 3)
  
  #scatter plot implementation
  output$scatter_plot <- plotly::renderPlotly({
    
    req(processed_data(), uploaded_data())
    
    scatter_df <- processed_data() %>%
      left_join(uploaded_data(), by = "Pharmaceutical") %>%
      filter(!is.na(Reported_mass_load),
             Reported_mass_load > 0,
             `Predicted Mass Load (ug/capita/day)` > 0) %>%
      mutate(
        Log10_Predicted = log10(`Predicted Mass Load (ug/capita/day)`),
        Log10_Reported = log10(Reported_mass_load),
        Log10_Ratio = Log10_Predicted - Log10_Reported,
        Category = case_when(
          abs(Log10_Ratio) < 1 ~ "Within 10×",
          Log10_Ratio >= 1 ~ "Predicted > 10× Reported",
          Log10_Ratio <= -1 ~ "Reported > 10× Predicted"
        ),
        
        # Tooltip text
        tooltip = paste0(
          "<b>", Pharmaceutical, "</b><br>",
          "Predicted: ", signif(`Predicted Mass Load (ug/capita/day)`, 4), "<br>",
          "Reported: ", signif(Reported_mass_load, 4), "<br>",
          "Ratio: ", round(10^Log10_Ratio, 3)
        )
      )
    
    p <- ggplot(scatter_df, aes(
      x = `Predicted Mass Load (ug/capita/day)`,
      y = Reported_mass_load,
      text = tooltip
    )) +
      geom_point(
        aes(color = Category, shape = Category),
        size = 3,
        alpha = 0.8
      ) +
      
      geom_abline(slope = 1, intercept = 0, linetype = "solid") +
      geom_abline(slope = 1, intercept = log10(10), linetype = "dashed") +
      geom_abline(slope = 1, intercept = -log10(10), linetype = "dashed") +
      
      scale_x_log10() +
      scale_y_log10() +
      
      scale_color_manual(values = c(
        "Within 10×" = "gray50",
        "Predicted > 10× Reported" = "blue",
        "Reported > 10× Predicted" = "red"
      )) +
      
      scale_shape_manual(
        values = c(
          "Within 10×" = 16,               
          "Predicted > 10× Reported" = 25,  
          "Reported > 10× Predicted" = 17   
        )
      ) +
      
      labs(
        x = "Predicted Mass Load (ug/capita/day)",
        y = "Reported Mass Load (ug/capita/day)",
        color = "Agreement Category",
        title = "Reported vs Predicted Pharmaceutical Mass Loads",
        subtitle = "Solid line = 1:1 agreement, dashed lines = ±10×"
      ) +
      
      theme_minimal(base_size = 14) +
      theme(
        legend.position = "right",
        panel.grid.minor = element_blank()
      )
    
    plotly::ggplotly(p, tooltip = "text")
  })
  
  
  
  # --- 1. GENERATE THE DATA ---
  # This creates the full dataset based on the population_served input
  synthetic_data_full <- eventReactive(input$generate_synth, {
    
    # Helper to ensure percentages sum to 100
    validate_percent <- function(vec, name) {
      total <- sum(vec)
      if (total != 100) {
        showNotification(
          paste(name, "percentages must sum to 100 (currently =", total, ")"),
          type = "error"
        )
        return(FALSE)
      }
      return(TRUE)
    }
    
    # Group inputs together for tracking
    age_pcts <- c(input$pct_age1, input$pct_age2, input$pct_age3, input$pct_age4, 
                  input$pct_age5, input$pct_age6, input$pct_age7, input$pct_age8, input$pct_age9)
    
    income_pcts <- c(input$pct_income1, input$pct_income2, input$pct_income3, input$pct_income4, 
                     input$pct_income5, input$pct_income6, input$pct_income7, input$pct_income8, 
                     input$pct_income9, input$pct_income10, input$pct_income11, input$pct_income12, 
                     input$pct_income13, input$pct_income14, input$pct_income15, input$pct_income16)
    
    # Run validation checks
    req(
      validate_percent(c(input$pct_male, input$pct_female), "Gender"),
      validate_percent(c(input$pct_hispanic, input$pct_white, input$pct_asian, input$pct_black, input$pct_other), "Race"),
      validate_percent(age_pcts, "Age"),
      validate_percent(income_pcts, "Income"),
      validate_percent(c(input$pct_uninsured, input$pct_public, input$pct_private), "Insurance")
    )
    
    n <- input$population_served  
    
    # Show a progress bar for large sewershed sizes
    withProgress(message = 'Generating Synthetic Population...', value = 0.5, {
      
      gender <- sample(c("Male","Female"), size = n, replace = TRUE, 
                       prob = c(input$pct_male, input$pct_female) / 100)
      
      race <- sample(c("Hispanic","Non-Hispanic White","Non-Hispanic Asian","Non-Hispanic Black","Non-Hispanic Other"), size = n, replace = TRUE, 
                     prob = c(input$pct_hispanic, input$pct_white, input$pct_asian, input$pct_black, input$pct_other) / 100)
      
      parse_range <- function(range_str) {
        parts <- strsplit(range_str, "-")[[1]]
        c(as.numeric(parts[1]), as.numeric(parts[2]))
      }
      
      # Expanded Age Generation Array (1 to 9)
      age_ranges <- list(
        parse_range(input$age1_range), parse_range(input$age2_range), parse_range(input$age3_range),
        parse_range(input$age4_range), parse_range(input$age5_range), parse_range(input$age6_range),
        parse_range(input$age7_range), parse_range(input$age8_range), parse_range(input$age9_range)
      )
      age_probs <- age_pcts / 100
      age_group <- sample(1:9, size = n, replace = TRUE, prob = age_probs)
      age <- sapply(age_group, function(i) {
        range <- age_ranges[[i]]
        sample(seq(range[1], range[2]), 1)
      })
      
      # Expanded Income Generation Array (1 to 16)
      income_ranges <- list(
        parse_range(input$inc1_range), parse_range(input$inc2_range), parse_range(input$inc3_range),
        parse_range(input$inc4_range), parse_range(input$inc5_range), parse_range(input$inc6_range),
        parse_range(input$inc7_range), parse_range(input$inc8_range), parse_range(input$inc9_range),
        parse_range(input$inc10_range), parse_range(input$inc11_range), parse_range(input$inc12_range),
        parse_range(input$inc13_range), parse_range(input$inc14_range), parse_range(input$inc15_range),
        parse_range(input$inc16_range)
      )
      income_probs <- income_pcts / 100
      income_group <- sample(1:16, size = n, replace = TRUE, prob = income_probs)
      income <- sapply(income_group, function(i) {
        range <- income_ranges[[i]]
        sample(seq(range[1], range[2]), 1)
      })
      
      insurance <- sample(c("Uninsured","Public","Private"), size = n, replace = TRUE, 
                          prob = c(input$pct_uninsured, input$pct_public, input$pct_private) / 100)
      
      setProgress(1)
      
      data.frame(
        Gender = gender,
        Race = race,
        Age = age,
        Income = income,
        Insurance = insurance
      )
    })
  })
  
  # --- 2. SHOW PREVIEW ---
  output$synthetic_table <- renderTable({
    req(synthetic_data_full())
    head(synthetic_data_full(), 20)
  })
  
  # --- 3. ADD DOWNLOAD HANDLER ---
  output$download_synthetic_csv <- downloadHandler(
    filename = function() {
      paste0("synthetic_population_", input$population_served, "_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(synthetic_data_full(), file, row.names = FALSE)
    }
  )
  
}

# run the app
shinyApp(ui = ui, server = server)