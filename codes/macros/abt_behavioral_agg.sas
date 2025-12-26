%macro make_abt_behavioral(
	period = ,
	data_in = ,
	data_out = ,
	id = cid,
	max_length = 12,
	lengths = 3 6 9 12,
	missing_allowed = 1,
	arrears_due_threshold = 1,
	arrears_days_threshold = 15,
	good_days_low = 0,
	good_days_high = 15
);
	%local n_lengths len length first_period index first_index n_var_agr n_sagr;

	%assert_dset_exists(&data_in);
	%assert_vars_exist(&data_in, &id);

	%let n_var_agr=6;
	%let n_sagr=3;

	%log_banner(STEP, make_abt_behavioral period=&period max_length=&max_length);

/*	Build list of periods (YYYYMM) used to reference monthly-suffixed columns*/
	data work._np_periods;
		periodp = input("&period", yymmn6.);
		do i = 0 to &max_length - 1;
			period = put(intnx('month', periodp - i, 'end'), yymmn6.);
			output;
		end;
		keep period;
	run;

	proc sql noprint;
		select period
			into :periods separated by ' '
		from work._np_periods
		order by 1; /* ascending */
	quit;

	%let first_period = %scan(&periods, 1, %str( ));

	data _null_;
		index intck('month', input("&first_period", yymmn6.), input("&period", yymmn6.)) + 1;
		call symputx('index', index);
	run;

/*	Define base variables and descriptions*/
	%let var1=CMaxI_Days; %let des1=Maximum Customer days for Ins product;
	%let var2=CMaxI_Due;  %let des2=Maximum Customer due for Ins product;
	%let var3=CMaxC_Days; %let des3=Maximum Customer days for Css product;
	%let var4=CMaxC_Due;  %let des4=Maximum Customer due for Css product;
	%let var5=CMaxA_Days; %let des5=Maximum Customer days for all product;
	%let var6=CMaxA_Due;  %let des6=Maximum Customer due for all product;

	%let sagr1=Mean;
	%let sagr2=Max;
	%let sagr3=Min;

/*	Compute aggregates*/
	data &data_out;
		set &data_in;

		%let n_lengths = %sysfunc(countw(&lengths, %str( )));
		%do len = 1 %to &n_lengths;
			%let length = %scan(&lengths, &len, %str( ));
			%let first_index = %eval(&index - &length + 1);
			%if &first_index < 1 %then %let first_index = 1;

/*			Rolling Mean/Max/Min for each base var*/
			%do v = 1 %to &n_var_agr;
				%do a = 1 %to &n_sagr;
					agr&length._&&sagr&a.._&&var&v = &&sagr&a(
						%do i = &first_index %to &index;
							%let p = %scan(&periods,&i,%str( ));
							&&var&v.._&p,
						%end;
					.);

					_nmiss = nmiss(
						%do i = &first_index %to &index;
							%let p = %scan(&periods,&i,%str( ));
							&&var&v.._&p,
						%end;
					.);

					ags&length._&&sagr&a.._&&var&v = agr&length._&&sagr&a.._&&var&v;
					if _nmiss > &missing_allowed then agr&length._&&sagr&a.._&&var&v = .m;

					label
						ags&length._&&sagr&a.._&&var&v =
							"&&sagr&a.. calculated on last &length. months on &&des&v"
						agr&length._&&sagr&a.._&&var&v =
							"&&sagr&a.. calculated on last &length. months on unmissing &&des&v";
				%end;
			%end;

/*			Counters*/
			act&length._n_arrears = sum(
				%do i = &first_index %to &index;
					%let p = %scan(&periods,&i,%str( ));
					(CMaxA_Due_&p >= &arrears_due_threshold),
				%end;
			.);
			label act&length._n_arrears = "Customer number of months in arrears on all loans";

			act&length._n_arrears_days = sum(
				%do i = &first_index %to &index;
					%let p = %scan(&periods,&i,%str( ));
					(CMaxA_Days_&p > &arrears_days_threshold),
				%end;
			.);
			label act&length._n_arrears_days = "Customer number of months with days past due > threshold on all loans";

			act&length._n_good_days = sum(
				%do i = &first_index %to &index;
					%let p = %scan(&periods,&i,%str( ));
					(&good_days_low < CMaxA_Days_&p and CMaxA_Days_&p < &good_days_high),
				%end;
			.);
			label act&length._n_good_days = "Customer number of months with 0<days past due<threshold on all loans";
		%end;

		drop _nmiss;
		keep &id agr: ags: act:;
	run;
%mend;

%macro make_abt(period);
	%make_abt_behavioral(
		period = &period,
		data_in = &data_wej,
		data_out = &data_wyj,
		id = &id_account,
		max_length = &max_length
	);
%mend;