## macros/transpose_utils.sas

+ Ensure input dataset exists and contains cid, period, days, due variables
+ Create %cmax_transpose ()
	+ For each client in each period, create CMax_days (days of delay) and CMax_due (count of months delay) of each period (periods as columns)


## macros/customer_features.sas

+ Create %build_customer_level
	+ Ensure transactions, cust_ids datasets exist
	+ Ensure cid, aid, product, period, fin_period, status, due_installments, paid_installments, n_installments, installment, income, spendings, leftn_installments exist in transactions
	+ Ensure cid exists in cust_ids
	+ Select clients with specific product in cust_ids until the period1 (_np_cus_&product)
	+ For each client, calculates
		+ Sums of paid_installments, n_installments, due_installments, installment
		+ Maximum of income, spendings, due_installments
		+ Number of loans with non-missing income
		+ Minimum of paid_installments, leftn_installments
	+ Calculates some derived rates:
		+ Utilization = Paid/total installments (how much of the schedule is paid)
		+ Due utilization = Due/total installments (share of schedule in arrears)
		+ Credit capacity = (installments + spendings)/ income (debt burden vs income)
	+ Select from _np_cus_&product:
		+ Seniority (months from loan start to period1)
		+ Total loan contracts of that product
		+ Total loan contracts of that product with status C (closed good) or B (closed bad)

## 00_config.sas

+ Assign project path, output folder paths
+ Ensure all folders exist before assigning libraries
+ Assign core parameters
	+ Target variable: default12
	+ Output library for variable definitions and stats
	+ Code directory
	+ Report directory

+ Create %power () and %powerc ()
+ Create %Additional_variables (app_IGJM) ?
+ Create %log_banner (print timestamp, messages)
+ Create %assert_dset_exists to ensure a dataset exists
+ Create %assert_vars_exist to ensure variables exist
+ Create %assert_sorted_by to sort a dataset

## 00A_build_abt_app.sas


+ Create %make_abt_behavioral (From a customer_level dataset with one row per `cid` and monthly columns of max days/due (ins, css, all), compute rolling summaries over the last 3, 6, 9, 12 months ending at period, plus arrears/good+days counters)
	+ Ensure &data_in, &id in &data_in exist
	+ Assign n_var_agr=6, n_sagr=3
	+ Select a list of period from current period + max_length, one row per period (_np_periods)
	+ Concat them into a list separated by ' ' order by asc
	+ Select first period (oldest month)
	+ Create (CMaxI_Days, CMaxI_Due, CMaxC_Days, CMaxC_Due, CMaxA_Days, CMaxA_Due) + base variables and (sagr1, sagr2, sagr3 + Mean, max, min) + aggregates
	+ For each row in data (one per customer), generates:
		+ For each window length (3, 6, 9, 12)
			+ For each of 6 base variables and 3 aggregates:
				+ Create aggregates&length_aggregatesname_basevariables (agr3_Mean_CMaxI_Days)
				+ Create act&length_n_arrears + Count of months the client had at least one unpaid installment (act3_n_arrears)
				+ Create act&length_n_arrears_days + Count of months with serious delinquency (act3_n_arrears_days)
				+ Create act&length_n_good_days + Count of months with mild delay (act3_n_good_days)

+ Create %_init_history_if_missing to ensure hist.transactions, hist.decisions exist


+ Create %build_abt_one_month (build application-level abt for one month using info as of the previous month)

| Step | What it does | Example output |
| :---: | :--- |:--- |
| 1 | Take applications for 202402 from pot.Production | _np_month_prod (rows with period = 202402) |
| 2 | Distinct customers in that month | _np_cust_uni (e.g., C001, C002) |
| 3 | Flag: had active loan in 202401 | _np_cust_uni_active (cid + act_cus_active = 1) |
| 4 | Customer-level features for INS and CSS as of 202401 | _np_cus_ins_hist/_agr, _np_cus_css_hist/_agr (utilization, CC, seniority, etc.) |
| 5 | All loans in 202402 (history + new apps), add time | _np_cus_all |
| 6 | Per application: loan counts and credit capacity at that moment in month | _np_cus_nloan (act_call_cc, act_cins_n_loan, act_ccss_n_loan, act_call_n_loan by aid) |
| 7 | Behavioral window start (e.g., 14 months before 202401) | proc_periodf (e.g., 202211) |
| 8 | Long table (cid, period, product, days, due) in window | _np_abt_tmp_cus |
| 9 | Max days/due by (cid, period), transpose to wide (INS, CSS, ALL) | _np_cmaxi/_cmaxc/_cmaxa_days and _due |
| 10 | Merge max tables by cid | _np_abt_beh (one row per cid, e.g., cmaxi_days_202312, …) |
| 11 | Rolling 3/6/9/12M Mean/Max/Min + n_arrears, n_arrears_days, n_good_days | _np_abt_beh_fin |
| 12 | Merge production + per-app loan/CC by aid | _np_abt_base |
| 13 | Final merge: base + INS/CSS hist + active flag + behavioral by cid | &out_abt (e.g., abt.abt_202402, one row per application with all features) |

	+ Ensure pot.Production & pot.transactions exist
	+ Create hist.transaction if not exists
	+ Select all rows with the target period in pot.Production (_np_month_prod)
	+ Select all clients in the target period
	+ Select all clients with active status in last period (period1) from hist.transactions (_np_cust_uni_active)
	+ Create customer_level for ins and css product (_np_cus_ins_hist, _np_cus_ins_agr, _np_cus_css_hist, _np_cus_css_agr)
	+ Count all loans in target period with active status (_np_cus_all)
	+ For each row (loan/app), calculates customer credit capacity, actual number of ins, css, all product
	+ Select cid, period, product, pay_days+15, due_installments from hist.transactions and cid in _np_cust_uni with period within target period and target period - max_length - 2
	+ Transpose them to make aggregations and periods into columns and one row per one client
	+ 



+ Create %build_abt_app ()	