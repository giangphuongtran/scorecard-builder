/* Build raw ABT

Outputs:
	- abt.abt_YYYYMM per month
*/
%macro _init_history_if_missing(
	hist_transactions = hist.transactions,
	hist_decisions = hist.decisions,
	pot_transactions = pot.transactions
);
	%if not %sysfunc(exist(&hist_transactions)) %then %do;
		%log_banner(INIT, Creating empty history transactions table: &hist_transactions);
		data &hist_transactions;
			length cid $10 aid $16 product $3 period fin_period $6 status $1;
			length due_installments paid_installments pay_days n_installments installments spendings income leftn_installments 8;
			stop;
		run;
	%end;

	%if not %sysfunc(exist(&hist_decisions)) %then %do;
		%log_banner(INIT, Creating empty history decisions table: &hist_decisions);
		data &hist_decisions;
			length cid $10 aid $16 product $3 period $6 decisions $1 decline_reason $20;
			length app_loan_amount app_n_installments pd cross_pd pr 8;
			format pd cross_pd pr nlpct12.2;
			stop;
		run;
	%end;
%mend;

%macro build_abt_one_month(
	proc_period = ,
	proc_period1 = ,
	pot_production = pot.Production,
	pot_transactions = pot.transactions,
	hist_transactions = hist.transactions,
	out_abt = ,
	max_length = 12
);
	%local proc_periodf;

	%assert_dset_exists(&pot_production);
	%assert_dset_exists(&pot_transactions);
	%_init_history_if_missing(hist_transactions = &hist_transactions);

	%log_banner(STEP1, Build ABT for &proc_period (prev = &proc_period1));

/*	Current month applications*/
	proc sql;
		create table work._np_month_prod as
		select *
		from &pot_production
		where period = "&proc_period";
	quit;

/*	Customers present in current month*/
	proc sql;
		create table work._np_cust_uni as
		select distinct cid
		from work._np_month_prod;
	quit;

/*	Active previous month flag*/
	proc sql;
		create table work._np_cust_uni_active as
		select distinct cid,
			1 as act_cus_active label="Customer had active (status = A) loans one month before"
		from &hist_transactions
		where period = "&proc_period1" and status = 'A';
	quit;

/*	Customer-level historical features*/
	%build_customer_level(
		transactions = &hist_transactions,
		cust_ids = work._np_cust_uni,
		proc_period1 = &proc_period1,
		product = ins,
		out_hist = work._np_cus_ins_hist,
		out_agr = work._np_cus_ins_agr
	);

	%build_customer_level(
		transactions = &hist_transactions,
		cust_ids = work._np_cust_uni,
		proc_period1 = &proc_period1,
		product = css,
		out_hist = work._np_cus_css_hist,
		out_agr = work._np_cus_css_agr
	);

/*	Loan counts and "within-month" cumulative CC*/
	proc sql;
		create table work._np_cus_all as
		select *
		from &hist_transactions
		where cid in (select cid from work._np_cust_uni)
			and period = "&proc_period"
			and status = 'A';
	quit;

	data work._np_cus_all;
		set work._np_cus_all work._np_month_prod(rename=(
			app_installments = installment
			app_spendings = spendings
			app_income = income
		));
		time = substr(aid, 4, 8);
	run;

	proc sort data=work._np_cus_all;
		by cid time aid;
	run;

	data work._np_cus_nloan;
		set work._np_cus_all;
		by cid;
		if first.cid then do;
			installment_cum = 0;
			n_all = 0; n_ins = 0; n_css = 0;
		end;
		installment_cum + installment;
		if product = 'ins' then n_ins + 1;
		if product = 'css' then n_css + 1;
		n_all + 1;

		act_call_cc = (installment_cum + spendings) / income;
		act_cins_n_loan = n_ins;
		act_ccss_n_loan = n_css;
		act_call_n_loan = n_all;

		label
			act_call_cc = "Customer credit capacity (all installments plus spendings) over income"
			act_cins_n_loan = "Actual customer loan number of Ins product"
			act_ccss_n_loan = "Actual customer loan number of Css product"
			act_call_n_loan = "Actual customer loan number"
		keep aid cid act_call: act_cins: act_ccss:;
	run;
	
	proc sort data=work._np_cus_nloan;
		by aid;
	run;

/*	Behavioral "wide" base: max days/due per cid per period, transposed*/
	data _null_;
		proc_periodf = put(
		intnx('month', input("&proc_period1", yymmn6.), - &max_length - 2, 'end'),
		yymmn6.
		);
		call symputx('proc_periodf', proc_periodf);
	run;

	proc sql;
		create table work._np_abt_tmp_cus as
		select cid, period, product,
			pay_days + 15 as days,
			due_installments as due
		from &hist_transactions
		where cid in (select cid from work._np_cust_uni)
			and "&proc_periodf" <= period <= "&proc_period1";
	quit;

	%cmax_transpose(
		in_dset = work._np_abt_tmp_cus,
		where = %str(product = 'ins'),
		out_prefix = cmaxi,
		out_days = work._np_cmaxi_days,
		out_due = work._np_cmaxi_due
	);
	%cmax_transpose(
		in_dset = work._np_abt_tmp_cus,
		where = %str(product = 'css'),
		out_prefix = cmaxc,
		out_days = work._np_cmaxc_days,
		out_due = work._np_cmaxc_due
	);
	%cmax_transpose(
		in_dset = work._np_abt_tmp_cus,
		where = %str(1=1),
		out_prefix = cmaxa,
		out_days = work._np_cmaxa_days,
		out_due = work._np_cmaxa_due
	);
	
	data work._np_abt_beh;
		merge
			work._np_cmaxa_days work._np_cmaxa_due
			work._np_cmaxi_days work._np_cmaxi_due
			work._np_cmaxc_days work._np_cmaxc_due
		by cid;
		if not missing(cid);
	run;

	%make_abt_behavioral(
		period = &proc_period1,
		data_in = work._np_abt_beh,
		data_out = work._np_abt_beh_fin,
		id = cid,
		max_length = &max_length
	);

/*	Assemblbe monthly ABT*/
	proc sort data=work._np_month_prod;
		by aid;
	run;

	data work._np_abt_base;
		merge work._np_month_prod(in=z) work._np_cus_nloan;
		by aid;
		if z;
	run;

	proc sort data=work._np_abt_base;
		by cid;
	run;
	proc sort data=work._np_abt_beh_fin;
		by cid;
	run;

	data &out_abt;
		merge
			work._np_abt_base(in=z)
			work._np_cus_ins_hist work._np_cus_ins_agr
			work._np_cus_css_hist work._np_cus_css_agr
			work._np_cust_uni_active
			work._np_abt_beh_fin;
		by cid;
		if z;
	run;
%mend;

%macro build_abt_monthly_from_potential(
	pot_production = pot.Production,
	pot_transactions = pot.transactions,
	hist_transactions = hist.transactions,
	out_abt_lib = abt,
	max_length = 12
);
	%local prod_periods n_prod_periods n_month proc_period proc_period1;

	%assert_dset_exists(&pot_production);
	%_init_history_if_missing(hist_transactions = &hist_transactions);

	proc sql noprint;
		select distinct period into :prod_periods separated by '#'
		from &pot_production
		order by period;
	quit;
	%let n_prod_periods = &sqlobs;

	%if &n_prod_periods < 2 %then %do;
		%put ERROR: Need at least 2 periods in &pot_production to build monthly ABTs (needs prev month).
		%abort cancel;
	%end;

/*	Bootstrap history with first month's transactions*/
	%let proc_period = %scan(&prod_periods, 1, #);
	%log_banner(INIT, Bootstrapping history.transactions with month &proc_period);
	proc sql;
		create table work._np_first_month_trans as
		select * from &pot_transactions
		where fin_period = "&proc_period";
	quit;
	proc append base = &hist_transactions data=work._np_first_month_trans force;
	run;

	%do n_month = 2 %to &n_prod_periods;
		%let proc_period = %scan(&prod_periods, &n_month, #);
		%let proc_period1 = %scan(&prod_periods, %eval(&n_month - 1), #);

		%build_abt_one_month(
			proc_period = &proc_period,
			proc_period1 = &proc_period1,
			pot_production = &pot_production,
			pot_transactions = &pot_transactions,
			hist_transactions = &hist_transactions,
			out_abt = &out_abt_lib..abt_&proc_period,
			max_length = &max_length
		);

		proc sql;
			create table work._np_month_trans as
			select * from &pot_transactions
			where fin_period = "&proc_period";
		quit;
		proc append base = &hist_transactions data=work._np_month_trans force;
		run;
	%end;
%mend;