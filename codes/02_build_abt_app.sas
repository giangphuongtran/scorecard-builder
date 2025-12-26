/* Build abt_app (training mart = features + targets)

Output:
	- mart.abt_app (one row per application/aid)
*/

%macro build_abt_app(
	abt_lib = abt,
	decisions_dset = hist.decisions,
	pot_default = pot.default,
	out_mart = mart.abt_app,
	response_condition = %str(Decision = 'A' and product = 'css'),
	response_n_months = 6
);
	%local abt_lib_uc abt_tables n_abt_tables res_periods n_res_periods first_period;

	%assert_dset_exists(&decisions_dset);
	%assert_dset_exists(&pot_default);

	%let abt_lib_uc = %upcase(&abt_lib);

	%log_banner(STEP2, Build training mart &out_mart from &abt_lib_uc..ABT_* plus targets);

/*	Stack all monthly ABTs from abt_lib into one dataset*/
	proc sql noprint;
		select cats("&abt_lib..", memname)
			into :abt_tables separated by ' '
		from dictionary.tables
		where libname = "&abt_lib_uc"
			and upcase(memname) like 'ABT_%'
		order by memname;
	quit;
	%let n_abt_tables = &sqlobs;

	%if &n_abt_tables = 0 %then %do;
		%put ERROR: No ABT tables found in library &abt_lib_uc (expected ABT_*).
		%abort cancel;
	%end;

	data work._np_abt_all;
		set &abt_tables;
	run;

	proc sort data=work._np_abt_all;
		by aid;
	run;

/*	Attach decision flag*/
	proc sort data=&decisions_dset out=work._np_decisions_keep(keep=aid decision);
		by aid;
	run;
	
	data work._np_abt_decision;
		merge work._np_abt_all(in=z) work._np_decisions_keep;
		by aid;
		if z;
	run;

/*	Build response metrix*/
	proc sort data=work._np_abt_decision(keep=cid aid period decision product)
		out=work._np_res(keep=cid aid period)
		by cid period aid;
		where &response_condition;
	run;

	proc sort data=work._np_res nodupkey;
		by cid period;
	run;

	proc transpose data=work._np_res out=work._np_response(drop=_name_ _label_) prefix=res_;
		by cid;
		id period;
		var aid;
	run;

	proc sql noprint;
		select distinct cats('res_', strip(period))
			into :res_periods separated by ' '
		from &pot_default
		order by period;
	quit;
	%let n_res_periods = &sqlobs;
	
	%if &n_res_periods = 0 %then %do;
		%put ERROR: Could not derive res_periods from &pot_default (expected period column);
		%abort cancel;
	%end;

	%let first_period = %substr(%scan(&res_periods, 1, %str( )), 5);

/*	Calculate cross-response label (future response within N months)*/
	proc sort data=work._np_abt_decision out=work._np_prod(keep=cid aid period);
		by cid;
	run;

	proc sort data=work._np_response;
		by cid;
	run;

	data work._np_response_cal;
		length cross_aid &res_periods $16 cross_response 8;
		array res_aid(&n_res_periods) &res_periods;

		merge work._np_prod(in=z) work._np_response;
		by cid;
		if z;
	
		index = intck('month', input("&first_period", yymmn6.), input("&period", yymmn6.)) + 2;
		max_index = index + &response_n_months - 2;
		cross_aid = '';
		cross_response = 0;
		cross_after_months = .;
		
		if 1 <= index <= &n_res_periods and 1 <= max_index <= &n_res_periods then do;
			do i = max_index to index by -1;
				if not missing(res_aid(i)) then do;
					cross_response = 1;
					cross_aid = res_aid(i);
					cross_after_months = i - index + 1;
				end;
			end;
		end;

		keep aid cid period cross_aid cross_response cross_after_months;
	run;

/*	Attach targets: defaults for this aid and for cross_aid*/
	proc sort data=work._np_response_cal;
		by aid;
	run;

	data work._np_response_cal2;
		merge
			work._np_response_cal(in=z)
			&pot_default(keep=aid default:);
		by aid;
		if z;
	run;

	proc sort data=work._np_response_cal2;
		by cross_aid;
	run;

	proc sort data=work._np_abt_decision out=work._np_decision_for_cross (keep=aid app_loan_amount app_n_installments);
		by aid;
	run;

	data work._np_response_cal3;
		merge
			work._np_response_cal2(in=z)
			&pot_default(
				keep=aid default:
				rename=(aid = cross_aid
				default3 = default_cross3
				default6 = default_cross6
				default9 = default_cross9
				default12 = default_cross12
			)
		)
			work._np_decision_for_cross(
				rename=(aid = cross_aid
					app_loan_amount = cross_app_loan_amount
					app_n_installments = cross_app_n_installments
				)
			);
		by cross_aid;
		if z;
	run;

/*	Final mart: ABT features + response/default labels*/
	proc sort data=work._np_abt_decision;
		by aid;
	run;

	proc sort data=work._np_response_cal3;
		by aid;
	run;

	data &out_mart;
		merge work._np_abt_decision(in=z) work._np_response_cal3;
		by aid;
		if z;
	run;
%mend;

