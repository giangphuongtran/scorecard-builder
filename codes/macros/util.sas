%macro log_banner(stage, msg);
	%put NOTE - [&stage] &msg;
	%put NOTE - Timestamp: %sysfunc(datetime(), datetime19.);
%mend;

%macro assert_dset_exists(dset);
	%if not %sysfunc(exist(&dset)) %then %do;
		%put ERROR: Required dataset does not exist: &dset;
		%abort cancel;
	%end;
%mend;

%macro assert_vars_exist(dset, vars);
	%local dsid i var rc;
	%let dsid = %sysfunc(open(&dset, i));
	%if &dsid = 0 %then %do;
		%put ERROR: Could not open dataset: &dset;
		%abort cancel;
	%end;
	%let i = 1;
	%do %while(%scan(&vars, &i, %str( )) ne );
		%let var = %scan(&vars, &i, %str( ));
		%if %sysfunc(varnum(&dsid, &var)) = 0 %then %do;
			%let rc = %sysfunc(close(&dsid));
			%put ERROR: Dataset &dset missing required variable: &var;
			%abort cancel;
		%end;
		%let i = %eval(&i + 1);
	%end;
	%let rc = %sysfunc(close(&dsid));
%mend;

%macro assert_sorted_by(dset, by);
	%put NOTE: (Info) Expecting &dset to be sorted by: &by;
%mend;