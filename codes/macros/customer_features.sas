%macro build_customer_level(
	transactions = ,
	cust_ids = ,
	proc_period1 = ,
	product = ,
	out_hist = ,
	out_agr = 
);
	%assert_dset_exists(&transactions);
	%assert_dset_exists(&cust_ids);
	%assert_vars_exist(&transactions, cid aid product period fin_period status due_installments paid_installments n_installments installment income spendings leftn_installments);
	%assert_vars_exist(&cust_ids, cid);

	%log_banner(STEP, build_customer_level product = &product period = &proc_period1);

	proc sql;
		create table work._np_cus_&product as
		select t.*
		from &transactions t
		where t.cid in (select cid from &cust_ids)
			and t.period <= "&proc_period1"
			and t.product = "&product";
	quit;

	proc means data=work._np_cus_&product nway noprint;
		class cid;
		var paid_installments n_installments leftn_installments
			due_installments income spendings installment;
		output out=work._np_cus_&product._agr0(drop=_type_ _freq_)
			sum(paid_installments n_installments due_installments installment) =
				paid_installments n_installments due_installments installment
			max(income spendings) = income spendings
			n(income) = act_c&product._n_loans_act
			max(due_installments) = act_c&product._maxdue
			min(paid_installments) = act_c&product._min_pninst
			min(leftn_installments) = act_c&product._min_lninst;
		where period = "&proc_period1";
	run;

	data &out_agr;
		set work._np_cus_&product._agr0;
		act_c&product._utl = paid_installments / n_installments;
		act_c&product._dueutl = due_installments / n_installments;
		act_c&product._cc = (installment + spendings) / income;
		keep cid act:;
		label
			act_c&product._utl = "Customer actual utilization rate on product &product"
			act_c&product._dueutl = "Customer due installments over all installments rate on product &product"
			act_c&product._cc = "Customer credit capacity (installment plus spendings) over income on product &product"
			act_c&product._maxdue = "Customer actual maximal due installments on product &product"
			act_c&product._min_pninst = "Customer minimal number of paid installments on product &product"
			act_c&product._min_lninst = "Customer minimal number of left installments on product &product"
			act_c&product._n_loans_act = "Customer actual number of loans on product &product";
  	run;

	proc sort data=&out_agr;
		by cid;
	run;

/*	History features up to proc_period1*/
	proc sql;
		create table &out_hist as
		select
			cid,
			max(intck('month', input(fin_period, yymmn6.), input("&proc_period1", yymmn6.)) + 1)
				as act_c&product._seniority label = "Customer seniority on product &product",
			min(intck('month', input(fin_period, yymmn6.), input("&proc_period1", yymmn6.)) + 1)
				as act_c&product._min_seniority label = "Customer minimal seniority on product &product",
			count(distinct aid) as act_c&product._n_loans_hist label = "Customer historial number of loans on product &product",
			sum((status = 'C')) as act_c&product._n_statC label = "Customer historical number of finished loans with status C on product &product",
			sum((status = 'B')) as act_c&product._n_statB label = "Customer historical number of finished loans with status B on product &product"
		from work._np_cus_&product
		group by cid
		order by cid;
	quit;

	proc sort data=&out_hist;
		by cid;
	run;
%mend;