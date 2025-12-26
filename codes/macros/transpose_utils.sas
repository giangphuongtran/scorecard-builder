/*Transpose helpers*/

%macro cmax_transpose(
	in_dset = ,
	id_var = cid,
	period_var = period,
	days_var = days,
	due_var = due,
	where = 1=1,
	out_prefix = ,
	out_days = ,
	out_due = 
);

	%assert_dset_exists(&in_dset);
	%assert_vars_exist(&in_dset, &id_var &period_var &days_var &due_var);

	%log_banner(STEP, cmax_transpose &out_prefix);

	proc means data=&in_dset nway noprint;
		class &id_var &period_var;
		var &days_var &due_var;
		output out=work._cmax_agg(drop=_type_ _freq_)
			max(&days_var &due_var) = &out_prefix._days &out_prefix._due;
		where &where;
	run;

	proc sort data=work._cmax_agg;
		by &id_var;
	run;

	proc transpose data=work._cmax_agg prefix=&out_prefix._days_
		out=&out_days(drop=_name_ _label_);
		var &out_prefix._days;
		id &period_var;
		by &id_var;
	run;

	proc transpose data=work._cmax_agg prefix=&out_prefix._due_
		out=&out_due(drop=_name_ _label_);
		var &out_prefix._due;
		id &period_var;
		by &id_var;
	run;
%mend;