# new_process/new_sas — Macros & Steps Reference

Tightened notes for all macros and step-based logic. Step tables use the same style as `%build_abt_one_month`.

---

## macros/customer_features.sas

### %build_customer_level(transactions=, cust_ids=, proc_period1=, product=, out_hist=, out_agr=)

| Step | Action | Output / Note |
| :---: | --- | --- |
| 1 | Assert `transactions`, `cust_ids` exist; assert required vars in each | — |
| 2 | Filter transactions: cid in cust_ids, period ≤ proc_period1, product = &product | work._np_cus_&product |
| 3 | **Aggregates at proc_period1**: proc means by cid — sums, max(income, spendings), n(income), max(due), min(paid_inst), min(leftn_inst) | work._np_cus_&product._agr0 |
| 4 | Derived rates: utilization, dueutl, cc; keep cid act: | &out_agr |
| 5 | **History features**: by cid — seniority, min_seniority, n_loans_hist, n_statC, n_statB | &out_hist |

**Step 2 – Input (example):** *transactions* (subset of cols):

| cid | aid | product | period | fin_period | status | due_installments | paid_installments | n_installments | installment | income | spendings | leftn_installments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C001 | A001 | ins | 202401 | 202401 | A | 0 | 6 | 12 | 500 | 3000 | 800 | 6 |
| C001 | A002 | ins | 202312 | 202312 | C | 0 | 12 | 12 | 400 | 3000 | 600 | 0 |

*cust_ids:*

| cid |
| --- |
| C001 |
| C002 |

**Step 2 – Output (example):** *_np_cus_&product* — same structure as input, filtered rows.

**Step 4 – Output (example):** *&out_agr*

| cid | act_cins_utl | act_cins_dueutl | act_cins_cc | act_cins_maxdue | act_cins_min_pninst | act_cins_min_lninst | act_cins_n_loans_act |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C001 | 0.50 | 0 | 0.43 | 0 | 6 | 6 | 1 |
| C002 | 0.25 | 0.10 | 0.55 | 1 | 3 | 9 | 2 |

**Step 5 – Output (example):** *&out_hist*

| cid | act_cins_seniority | act_cins_min_seniority | act_cins_n_loans_hist | act_cins_n_statC | act_cins_n_statB |
| --- | --- | --- | --- | --- | --- |
| C001 | 5 | 1 | 2 | 1 | 0 |
| C002 | 3 | 3 | 1 | 0 | 0 |

---

## 00_config.sas

- **Paths**: PROJECT_ROOT, FOLDER_00–10 (raw_data, data_prep, variable_metadata, binning, woe, var_selection, model_building, model_assess, scorecard, scoring_code, reports); subdirs created via %_ensure_dir.
- **Libraries**: pot (raw/potential), inlib (data_prep), abt, out, freq, adj, models, registry, scorecard, reports.
- **Parameters**: tar=default12, lib=out, dir_codes, reportsdir, nodedir.
- **Macros**: %power, %powerc (power/Gini-style stats); %Additional_variables (app_IGJM, filter 197501–198712 css Decision='A'); %log_banner; %assert_dset_exists; %assert_vars_exist; %assert_sorted_by (note only).

### %_ensure_dir(path)

| Step | Action |
| :---: | --- |
| 1 | Create parent directory and leaf so path exists (dcreate). |

### %cmax_transpose(in_dset=, id_var=cid, period_var=period, days_var=days, due_var=due, where=1=1, out_prefix=, out_days=, out_due=)

| Step | Action | Example |
| :---: | --- | :--- |
| 1 | Assert dataset and vars (id, period, days, due) exist | — |
| 2 | Proc means nway by id_var period_var: max(days_var, due_var) → &out_prefix._days, &out_prefix._due | work._cmax_agg |
| 3 | Transpose _days by period → columns &out_prefix._days_YYYYMM | &out_days |
| 4 | Transpose _due by period → columns &out_prefix._due_YYYYMM | &out_due |

**Input (example):** *in_dset* — long layout, one row per (cid, period, product) after optional *where*:

| cid | period | product | days | due |
| --- | --- | --- | --- | --- |
| C001 | 202312 | ins | 20 | 2 |
| C001 | 202401 | css | 10 | 0 |
| C002 | 202401 | ins | 25 | 1 |

**Output (example):** *&out_days* — one row per cid, periods as columns (e.g. out_prefix=cmaxi):

| cid | cmaxi_days_202312 | cmaxi_days_202401 |
| --- | --- | --- |
| C001 | 20 | . |
| C002 | . | 25 |

*&out_due* — same layout for due:

| cid | cmaxi_due_202312 | cmaxi_due_202401 |
| --- | --- | --- |
| C001 | 2 | . |
| C002 | . | 1 |

---

## 00A_build_abt_app.sas

### %make_abt_behavioral(period=, data_in=, data_out=, id=cid, max_length=12, lengths=3 6 9 12, …)

| Step | Action | Example |
| :---: | --- | :--- |
| 1 | Assert data_in exists and contains &id | — |
| 2 | Build period list: period and (max_length-1) months back; order asc; first_period = oldest | _np_periods, &periods, &first_period |
| 3 | Index = position of reference month (1-based) in period list | &index |
| 4 | Base vars: CMaxI_Days/Due, CMaxC_Days/Due, CMaxA_Days/Due (6). Aggregates: Mean, Max, Min (3) | — |
| 5 | For each window length (e.g. 3,6,9,12): rolling Mean/Max/Min over base vars → agr&length._Mean_CMaxI_Days etc.; allow missing_allowed; set to .m if too many missing | &data_out |
| 6 | Same windows: act&length._n_arrears (months CMaxA_Due ≥ threshold), n_arrears_days (CMaxA_Days > days threshold), n_good_days (0 < days < high) | act3_n_arrears, act3_n_arrears_days, act3_n_good_days |

### %_init_history_if_missing(hist_transactions=, hist_decisions=, pot_transactions=)

| Step | Condition | Action |
| :---: | --- | --- |
| 1 | hist.transactions missing | Create empty: cid aid product period fin_period status due_installments paid_installments pay_days n_installments installment spendings income leftn_installments |
| 2 | hist.decisions missing | Create empty: cid aid product period decision decline_reason app_loan_amount app_n_installments pd cross_pd pr |

### %build_abt_one_month(proc_period=, proc_period1=, pot_production=, pot_transactions=, hist_transactions=, out_abt=, max_length=12)

| Step | What it does | Example output |
| :---: | --- | :--- |
| 1 | Applications for current month from pot.Production | _np_month_prod (period = proc_period) |
| 2 | Distinct cid in that month | _np_cust_uni |
| 3 | Flag: had active loan (status=A) in proc_period1 | _np_cust_uni_active (act_cus_active=1) |
| 4 | %build_customer_level for ins and css as of proc_period1 | _np_cus_ins_hist/_agr, _np_cus_css_hist/_agr |
| 5 | All loans in proc_period: hist (status=A) + month_prod (with app_*→installment/spendings/income), add time from aid | _np_cus_all |
| 6 | By aid: cumulative installment, n_ins/n_css/n_all, act_call_cc, act_cins_n_loan, act_ccss_n_loan, act_call_n_loan | _np_cus_nloan |
| 7 | Behavioral window start: proc_period1 − (max_length+2) months | proc_periodf |
| 8 | Long table cid, period, product, days (pay_days+15), due in [proc_periodf, proc_period1] | _np_abt_tmp_cus |
| 9 | %cmax_transpose for ins, css, all → wide days/due by period | _np_cmaxi/_cmaxc/_cmaxa_days, _due |
| 10 | Merge all cmax tables by cid | _np_abt_beh |
| 11 | %make_abt_behavioral(period=proc_period1) → rolling 3/6/9/12 + n_arrears, n_arrears_days, n_good_days | _np_abt_beh_fin |
| 12 | Merge month_prod + _np_cus_nloan by aid | _np_abt_base |
| 13 | Merge base + ins/css hist/agr + active + behavioral by cid, keep rows in base | &out_abt (e.g. abt.abt_YYYYMM) |

**Step 8 – Input:** hist.transactions (cid, period, product, pay_days, due_installments) in window. **Output (example):** *_np_abt_tmp_cus*

| cid | period | product | days | due |
| --- | --- | --- | --- | --- |
| C001 | 202312 | ins | 35 | 0 |
| C001 | 202401 | ins | 20 | 2 |
| C001 | 202401 | css | 10 | 0 |
| C002 | 202401 | ins | 25 | 1 |

**Step 10 – Output (example):** *_np_abt_beh* — one row per cid, wide period columns (e.g. cmaxi_days_202312, cmaxi_due_202312, cmaxc_*, cmaxa_*):

| cid | cmaxi_days_202312 | cmaxi_due_202312 | cmaxi_days_202401 | cmaxi_due_202401 | cmaxa_days_202312 | cmaxa_due_202312 | … |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C001 | 35 | 0 | 20 | 2 | 35 | 0 | … |
| C002 | . | . | 25 | 1 | 25 | 1 | … |

**Step 13 – Output (example):** *&out_abt* — one row per application (aid), all features merged by cid:

| aid | cid | period | product | act_call_cc | act_cins_n_loan | act_ccss_n_loan | act_cus_active | agr3_Mean_CMaxI_Days | act3_n_arrears | act_cins_utl | … |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A001 | C001 | 202402 | css | 0.45 | 1 | 0 | 1 | 18.5 | 1 | 0.50 | … |
| A002 | C002 | 202402 | ins | 0.38 | 0 | 1 | 0 | . | 0 | . | … |

### %build_abt_monthly_from_potential(pot_production=, pot_transactions=, hist_transactions=, out_abt_lib=abt, max_length=12)

| Step | What it does | Example |
| :---: | --- | :--- |
| 1 | Distinct periods from pot.Production, ordered | prod_periods, n_prod_periods |
| 2 | Require n_prod_periods ≥ 2 | Else abort |
| 3 | Bootstrap: append transactions with fin_period = first period to hist.transactions | So second month has prior history |
| 4 | Loop n_month = 2 to n_prod_periods: set proc_period, proc_period1; %build_abt_one_month → out_abt = &out_abt_lib..abt_&proc_period | abt.abt_202402, abt.abt_202403, … |
| 5 | After each month: append that month’s transactions (fin_period = proc_period) to hist.transactions | Next iteration sees cumulative history |

### %build_abt_app(abt_lib=, decisions_dset=, pot_production=, pot_default=, out_abt_app=, response_condition=, response_n_months=6)

| Step | What it does | Example |
| :---: | --- | :--- |
| 1 | List monthly ABT tables in abt_lib (ABT_1% or ABT_2%); stack into one dataset | abt (stacked) |
| 2 | Merge with decisions (aid, decision); rename decision→Decision | decision |
| 3 | Filter by response_condition; distinct cid×period; transpose by period → res_YYYYMM | response |
| 4 | Response period names from pot.Production | res_periods |
| 5 | Cross-response: for each (cid, period) find response in next response_n_months; cross_aid, cross_response, cross_after_monhs | response_cal |
| 6 | Attach pot.default (aid, default:) by aid | response_cal2 |
| 7 | Attach cross_aid defaults (renamed default_cross*) and decision app_* (renamed cross_app_*) by cross_aid | response_cal3 |
| 8 | Merge decision + response_cal3 by aid | &out_abt_app (e.g. inlib.abt_app) |

**Step 1 – Input (example):** *abt.abt_202401*, *abt.abt_202402*, … (one row per aid, period-specific, with app:, act:, agr:).

**Step 2 – Input:** *decisions_dset*

| aid | decision |
| --- | --- |
| A001 | A |
| A002 | D |
| A003 | A |

**Step 2 – Output (example):** *decision* — stacked abt + Decision (all aids from stacked ABT, decision merged).

**Step 3 – Output (example):** *response* — one row per cid, response period as columns (aid or blank):

| cid | res_202401 | res_202402 | res_202403 |
| --- | --- | --- | --- |
| C001 | A001 | A005 | |
| C002 | | A002 | A006 |

**Step 8 – Output (example):** *&out_abt_app* (inlib.abt_app) — one row per application, with response and default info:

| aid | cid | period | Decision | product | cross_aid | cross_response | cross_after_monhs | default12 | default_cross12 | app_loan_amount | … |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A001 | C001 | 202401 | A | css | A005 | 1 | 2 | 0 | 0 | 5000 | … |
| A002 | C002 | 202402 | D | ins | | 0 | . | . | . | 3000 | … |

---

## 01_train_valid.sas (no macro)

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | Read in_abt; %Additional_variables; filter tar in (0,1,.i,.d) and ranuni(1)<prop | — |
| 2 | Split: 40%→valid, 60%→train (ranuni); keep tar aid cid outstanding period default: app: act: | abt.train, abt.valid |

**Input (example):** *in_abt* (e.g. inlib.abt_app) — one row per application:

| aid | cid | period | product | tar | outstanding | default12 | app_loan_amount | act_call_cc | … |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A001 | C001 | 202401 | css | 0 | 5000 | 0 | 5000 | 0.45 | … |
| A002 | C002 | 202402 | ins | 1 | 3000 | 1 | 3000 | 0.38 | … |
| A003 | C003 | 202401 | css | .i | 0 | . | 4000 | 0.52 | … |

**Output (example):** *abt.train* / *abt.valid* — same columns, subset of rows (tar in 0,1; split by ranuni).

---

## 02_labels.sas (no macro)

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | From dictionary.columns (lib=ABT, mem=TRAIN): name, label | inlib.labels |

**Output (example):** *inlib.labels*

| name | label |
| --- | --- |
| aid | Application ID |
| act_call_cc | Customer credit capacity (all installments plus spendings) over income |
| agr3_Mean_CMaxI_Days | Mean calculated on last 3 months on Maximum Customer days for Ins product |
| APP_LOAN_AMOUNT | Requested loan amount |

---

## 03_variable_definition.sas

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | From zb (abt.train): numeric names like AGR%, ACT%, AGS%, APP% → type=int | &lib..variable_definition |
| 2 | Same, char → type=nom; var_list, var_count | nom, &var_list |
| 3 | %count_distinct: for each nom var, count distinct in zb | uni |
| 4 | Insert into variable_definition: nom vars with 2 ≤ distinct ≤ 200 as type nom | &lib..variable_definition |
| 5 | Sort by variable | — |

**Output (example):** *&lib..variable_definition*

| variable | type | ver |
| --- | --- | --- |
| ACT_CALL_CC | int | y |
| AGR3_Mean_CMaxI_Days | int | y |
| APP_LOAN_AMOUNT | int | y |
| APP_CHAR_JOB_CODE | nom | y |
| APP_CHAR_GENDER | nom | y |

*Macro %count_distinct: loop over &var_list, proc sql count(distinct &var_name) from &zb → uni.*

---

## 04_bining_nominal.sas

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | Get nominal vars from data_variableset; min_count = min_percent% of zb n | var_to_join, min_count |
| 2 | For each nominal var: means by var, mean(tar)=br, freq>&min_count; cluster Ward; tree nclusters=&max_n_splitting_points | podz (cluster assignment) |
| 3 | %to_encode: build condition per cluster (when (val1,val2,...)), grp, variable | podz_nom |
| 4 | Append to t.bining_nominal; drop rows where condition like '%&cl%' | t.bining_nominal (freq lib) |

**Output (example):** *t.bining_nominal* — one row per (variable, bin):

| variable | condition | grp |
| --- | --- | --- |
| APP_CHAR_JOB_CODE | when ('E','F','G') | 1 |
| APP_CHAR_JOB_CODE | when ('A','B','C') | 2 |
| APP_CHAR_JOB_CODE | when ('D') | 3 |
| APP_CHAR_GENDER | when ('M') | 1 |
| APP_CHAR_GENDER | when ('F') | 2 |

*Parameters: zb, data_variableset, max_n_splitting_points, min_percent (main.sas).*

---

## 05_bining_nominal_without_joining.sas

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | Nominal vars from data_variableset; min_count as in 04 | var_without_join |
| 2 | For each var: means by var, freq>&min_count; no cluster — each category = cluster (cluster=_n_) | podz |
| 3 | Same encode: condition when (…), grp, variable | podz_nom |
| 4 | Append to t.bining_nominal_wj | t.bining_nominal_wj |

**Output (example):** *t.bining_nominal_wj* — same structure as 04 (variable, condition, grp); each category is its own grp.

---

## 06_tree.sas (interval binning)

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | Interval vars from data_variableset; min_count from zb | int_ord_var, min_count |
| 2 | %create_bins(var): trim var (p1, p99); tstat = class var, sum(tar), n; initial condition "not missing(&var)" | conditions |
| 3 | %step(cond_num): from conditions compute count, bad_count, good_count; scan cutpoints; maximize Gini (g) or entropy (h); split into two conditions, can_split=0/1 | conditions updated |
| 4 | %splits: repeat up to max_n_splitting_points: pick condition with can_split=1 (depth desc, criterion), %step | conditions (all splits) |
| 5 | %for_all_vars: run %create_bins for each interval var; append conditions to all_splits | all_splits |
| 6 | Keep variable, condition, grp (obs_num) | t.bining_int_nonmon |

**Output (example):** *t.bining_int_nonmon*

| variable | condition | grp |
| --- | --- | --- |
| ACT_CALL_CC | not missing(ACT_CALL_CC) and ACT_CALL_CC <= 0.35 | 1 |
| ACT_CALL_CC | 0.35 < ACT_CALL_CC <= 0.55 | 2 |
| ACT_CALL_CC | 0.55 < ACT_CALL_CC | 3 |
| AGR3_Mean_CMaxI_Days | not missing(AGR3_Mean_CMaxI_Days) and AGR3_Mean_CMaxI_Days <= 10 | 1 |
| AGR3_Mean_CMaxI_Days | 10 < AGR3_Mean_CMaxI_Days <= 25 | 2 |

*Parameters: zb, data_variableset, max_n_splitting_points, min_percent. Criterion: &crit (h or g).*

---

## 07_coding.sas (WOE coding)

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | Sort Bining_int → interval_splits_sorted; add “missing” bin per variable → &lib..bining_interval_fin | bining_interval_fin |
| 2 | Copy bining_nominal → &lib..bining_nominal_fin | bining_nominal_fin |
| 3 | Combine interval + nominal; apply adj.* tables (exclude by adj names); add “otherwise” per variable | splits |
| 4 | Generate coding_code_tmp.sas: select/when → GRP_&var; run grp; proc means → n_bads_cat, n_cat, n_ind_cat | counts |
| 5 | Merge splits + counts; WOE, br, logit, Percent*, wi, ivi; cap br; drop small percent; renumber grp; sort by variable, otherwise_ind, order_tar, br | &lib..scorecard_all |
| 6 | Generate coding_code.sas: assign GRP_&var and WOE_&var from transformed (logit); %include for zb and zb_v | abt.train_woe, abt.valid_woe |

**Step 3 – Output (example):** *splits* — interval + nominal + otherwise:

| variable | condition | grp |
| --- | --- | --- |
| ACT_CALL_CC | not missing(ACT_CALL_CC) and ACT_CALL_CC <= 0.35 | 1 |
| ACT_CALL_CC | 0.35 < ACT_CALL_CC <= 0.55 | 2 |
| ACT_CALL_CC | otherwise | 3 |
| APP_CHAR_JOB_CODE | when ('E','F','G') | 1 |
| APP_CHAR_JOB_CODE | when ('A','B','C') | 2 |
| APP_CHAR_JOB_CODE | otherwise | 3 |

**Step 5 – Output (example):** *&lib..scorecard_all*

| variable | condition | grp | n_cat | n_bads_cat | n_goods_cat | woe | br | transformed | Percent | Percent_bads | Percent_goods |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT_CALL_CC | ... and ACT_CALL_CC <= 0.35 | 1 | 1200 | 60 | 1140 | -0.52 | 0.05 | -2.94 | 0.12 | 0.06 | 0.14 |
| ACT_CALL_CC | 0.35 < ACT_CALL_CC <= 0.55 | 2 | 3500 | 280 | 3220 | 0.10 | 0.08 | -2.31 | 0.35 | 0.28 | 0.36 |
| ACT_CALL_CC | otherwise | 3 | 5300 | 660 | 4640 | 0.45 | 0.12 | -2.00 | 0.53 | 0.66 | 0.50 |

**Step 6 – Output (example):** *abt.train_woe* / *abt.valid_woe* — same rows as zb/zb_v, with added GRP_* and WOE_* columns.

*Parameters: Bining_int, bining_nominal, zb, zb_v (main.sas).*

---

## 08_variable_pre_selection_1step.sas

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | Variables from Scorecard_all with max(grp)>1 | &lib..variables_inset |
| 2 | %validuj_zmienne(train, lista, max_il): for each variable, logistic tar=WOE_var; from Association get c → AR; append variable, ar_train | variables_stat_1step |
| 3 | Sort by descending AR_Train | &lib..variables_stat_1step |

**Step 1 – Output (example):** *&lib..variables_inset*

| name |
| --- |
| WOE_ACT_CALL_CC |
| WOE_AGR3_Mean_CMaxI_Days |
| WOE_APP_CHAR_JOB_CODE |

**Step 3 – Output (example):** *&lib..variables_stat_1step*

| variable | ar_train |
| --- | --- |
| WOE_ACT_CALL_CC | 0.42 |
| WOE_AGR3_Mean_CMaxI_Days | 0.38 |
| WOE_APP_CHAR_JOB_CODE | 0.25 |

---

## 09_variable_pre_selection_full.sas

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | Variables from good_variables_stat_1step (name = substr 5 of variable) | &lib..variables_inset_fin |
| 2 | %validuj_zmienne(train, valid, lista): for each variable compute many stats (org: pr_miss, pr_mfrequent, n_uni, c_01, ks_01, h_01, max_dist_01; WOE: ar_train/valid/diff; grp: h_grp, max_dist_grp, h_br_grp_tv, etc.) | variables_stat |
| 3 | Sort by descending AR_Train | &lib..variables_stat |

*Uses %calculate_org, %calculate_ar_woe, %calculate_h, %calculate_by_periods_org.*

---

## 10_variable_corrections.sas (no macro)

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | Define manual bins (e.g. ACT_AGE) in adj lib: variable, condition, grp | adj.ACT_AGE (example) |

*Used to override/merge with automatic binning in 07_coding (splits merge adj.*).*

---

## 11_variable_reports.sas

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | All_possible_variables.html from inlib.labels | reportsdir |
| 2 | Variables_stat + labels → All_variables.html (gini before/after, diff, PR_Miss, N_Uni, etc.) | — |
| 3 | Chosen_variables from Scorecard_all; karta with POP, GD, BD, IV, etc.; Varclus; clusters, ivi | karta, clusters |
| 4 | Chosen_variables.html, Clustered_variables.html, Cluster_reports.html | — |
| 5 | Rebuild abt with Additional_variables, quarter/year; apply coding_code.sas for chosen vars → abt_woe | abt, abt_woe |
| 6 | %make_details: per chosen var, detail HTML (karta, tabulate, gplot by year/period) | &var..html |

*Parameters: import_data, zb, lib, Chosen_variables, reportsdir (config/main).*

---

## 12_steps_selection.sas

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | Variables from Good_variables (WOE_* list) | &variables |
| 2 | %steps(method): logistic selection=&method (e.g. FORWARD SLSTAY= SLENTRY=); from ParameterEstimates build VariablesInModel, NumberOfVariables, method | steps |
| 3 | %poziomy: call %steps with chosen SLSTAY/SLENTRY | steps |
| 4 | %valid_list(steps, &lib..steps_models, train, valid): for each row run %calculate → AR, VIF, corr, ConIndex, WaldChiSq, ProbChiSq; append model | &lib..steps_models |

---

## 13_score_selection.sas

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | Model (variable list) from Steps_models(obs=1) | &variables |
| 2 | Logistic selection=score best=&best start=&start stop=&stop → bestsubsets | score |
| 3 | subset: SBC, AIC from score; method label | subset |
| 4 | %valid_list(subset, &lib..branch_models, train, valid) | &lib..branch_models |

---

## 14_expert_models.sas

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | Hard-code subset: method='Expert', VariablesInModel = fixed WOE list, NumberOfVariables=6 | subset |
| 2 | %valid_list(subset, &lib..model_expert, train, valid) | &lib..model_expert |

---

## 15_model_assessment.sas

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | Stack insets (e.g. Steps_models, Model_expert, Branch_models); add AR_Diff; sort by model, then by descending AR_Valid | &lib..all_models |
| 2 | Filter good_models: VIF, corr, ConIndex, |AR_Diff|, ProbChiSq, n_beta_minus=0 within thresholds | &lib..good_models |
| 3 | %validuj_dobre(lista=t good_models, train, valid, max_nr_mod): for each model get ParameterEstimates; build scorecard (betas, factor/offset, SCORECARD_POINTS); score train/valid; compute KS, H, SD, AR on score, lifts/gains, VIF/corr on valid; append to lista_valid | &lib..good_models_valid (concept) |

*Score scale: factor=20/log(2), offset=300 by default; optional custom scale (min_p, max_p).*

---

## 16_bootstrap_validation.sas

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | Read kal.score&the_best_model (tar, score_points), keep tar in (0,1) | score |
| 2 | %validuj: loop seed=1 to num_seed; surveyselect URS strata=tar, same n per stratum; npar1way KS; logistic tar=score_points → AR; append seed, ar, ks | kal.bootstrap&the_best_model |
| 3 | Proc means of bootstrap: mean, p50, min, max, cv, range, qrange, uclm, lclm | kal.cross_stat&the_best_model |

*Parameters: the_best_model, num_seed (main.sas).*

---

## 17_ci_gini.sas

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | Logistic tar=score_points on kal.score&the_best_model → ROC Association; c (AUC) | roc |
| 2 | Freq by score_points and tar; build contingency for C computation | freq, tfreq |
| 3 | Compute std_c, var_c, C, 95%/99% CI for C and AR (ar=2*c-1) via formula (SumDefaults, SumNonDefaults, Equal, Part1–4) | czesci |
| 4 | Output with labels | kal.ci_c_ar&the_best_model |

---

## 18_final_report.sas

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | Rebuild sc_train, sc_valid from import_data/import_validate using Scorecard_Scorecard&the_best_model (GRP lookup → SCORECARD_POINTS) | sc_train, sc_valid |
| 2 | Compute scale (min, max, range of score), KS on score, H/H_br, SD, max_dist, prop_sd_range, AR on score, VIF/corr/ConIndex train & valid, max_abs_sd_zm, prop_max_sd_range_zm | model (one row) |
| 3 | ODS HTML Report.html (frame): variables in model, variables_stat (quality, discriminant, stability), GRP histograms (org and grp), scorecard attributes, scorecard points table, scale, discriminant/stability of model, CI for C/AR, bootstrap univariate and charts, effects, collinearity, sd_zmienne | FOLDER_10_REPORTS |

*Macros: %przygotuj_histogramy, %przygotuj_histogramy_score, %przygotuj_histogramy_n, %przygotuj_histogramy_br, %wykresy_org, %wykresy_grp.*

---

## 19_scoring_code.sas

| Step | Action | Output |
| :---: | --- | :--- |
| 1 | Read models.Scorecard_Scorecard&the_best_model; sort by variable, otherwise_ind, order_tar, br | p |
| 2 | Write scoring_code.sas: data &zbior._score; set &zbior; select/when(condition) → SCORECARD_POINTS=sum(…, points); PSC_&var=points; otherwise same | &FOLDER_09_SCORING_CODE\scoring_code.sas |

**Step 1 – Input (example):** *models.Scorecard_Scorecard&the_best_model* (after rename: variable, condition, br, SCORECARD_POINTS):

| variable | condition | br | SCORECARD_POINTS |
| --- | --- | --- | --- |
| ACT_CALL_CC | not missing(ACT_CALL_CC) and ACT_CALL_CC <= 0.35 | 0.05 | 45 |
| ACT_CALL_CC | 0.35 < ACT_CALL_CC <= 0.55 | 0.08 | 35 |
| ACT_CALL_CC | otherwise | 0.12 | 20 |
| APP_CHAR_JOB_CODE | when ('E','F','G') | 0.04 | 50 |
| APP_CHAR_JOB_CODE | when ('A','B','C') | 0.10 | 30 |
| APP_CHAR_JOB_CODE | otherwise | 0.15 | 15 |

**Step 2 – Output (example):** *scoring_code.sas* — generated DATA step that, for each observation in &zbior, evaluates conditions and adds SCORECARD_POINTS (and PSC_&var per variable). **Runtime output:** *&zbior._score* — input rows plus columns SCORECARD_POINTS, PSC_ACT_CALL_CC, PSC_APP_CHAR_JOB_CODE, …

*Parameters: the_best_model, score_points, zb (main.sas).*

---

## main.sas (orchestration)

| Step | Action |
| :---: | --- |
| 1 | %include 00_config.sas |
| 2 | %include macros (util, transpose_utils, abt_behavioral_agg, customer_features) — note: in new_sas only customer_features is under macros; transpose may live in config elsewhere |
| 3 | %include 00A_build_abt_app.sas |
| 4 | %build_abt_monthly_from_potential → abt.abt_YYYYMM |
| 5 | %build_abt_app (or %build_abt_app_professor if alias) → inlib.abt_app |
| 6 | Set zb, zb_v, data_variableset, max_n_splitting_points, min_percent, prop |
| 7 | %include 01–19 in order (train_valid, labels, variable_definition, bining_nominal, bining_nominal_without_joining, tree, coding + variable_pre_selection_1step + good_variables filter + variable_pre_selection_full + good_variables filter, variable_corrections, calc_design, chosen_variables, variable_reports, steps_selection, score_selection, expert_models, model_assessment, bootstrap_validation, ci_gini, final_report, scoring_code) |

*Naming: 00A defines %build_abt_app; main.sas may call %build_abt_app_professor — ensure one macro name is used consistently.*
