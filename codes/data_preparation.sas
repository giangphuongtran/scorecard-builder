%let project_dir=\\10.200.43.8\Sandisk\credit-scoring;

/*libname raw "&project_dir\data\raw" compress=yes;*/
libname inlib "&project_dir\data\raw\inlib" compress=yes;
libname abt "&project_dir\data\work\abt" compress=yes;
libname models "&project_dir\models" compress=yes;
libname reports "&project_dir\reports" compress=yes;

%let input_data=inlib.abt_app;
%let target_var=default12;

proc contents data=&input_data;
	title "Step 1: Verify data exists";
run;

proc sql;
	select
		count(1) as total_observations,
		count(distinct aid) as unique_app,
		count(distinct cid) as unique_cust
	from &input_data;
quit;

proc freq data=&input_data;
	tables &target_var / missing nocum;
	title "Target variable distribution";
run;

proc freq data=&input_data;
	tables decision product period / missing;
	title "Decision, Product, Period Distribution";
run;

data work.application_data;
	set &input_data;
	where decision = 'A'
		and product = 'css'
		and '197501' <= period <= '198712'
		and &target_var in (0, 1);

	keep
		aid cid period
		&target_var
		decision product
		app: act: default:
	;
run;

proc freq data=work.application_data;
	tables &target_var / missing nocum;
	title "Target variable distribution";
run;

proc means data=work.application_data n;
	title "Sample Size";
run;

proc contents data=work.application_data out=vars(keep=name type) noprint;
run;

proc sql noprint;
	select name into :app_num_list separated by ' '
	from vars
	where upcase(name) like 'APP_%' and type=1;
quit;

proc sql noprint;
	select name into :app_char_list separated by ' '
	from vars
	where upcase(name) like 'APP_%' and type=2;
quit;

proc means data=work.application_data n nmiss mean std min max;
	var &app_num_list;
	title "Application Numeric Variables Summary";
run;

proc freq data=work.application_data;
	tables &app_char_list / missing;
	title "Application Character Variables";
run;

proc means data=work.application_data n nmiss mean std min max;
	var act:;
	title "Behavioral Variables Summary";
run;

proc means data=work.application_data n nmiss;
	var _numeric_ ;
	title "Missing Values Check for all numeric variables";
run;

data abt.train abt.valid;
	set work.application_data;

	if ranuni(12345) < 0.7 then output abt.train;
	else output abt.valid;
run;

proc sql;
	select
		'Train' as dataset,
		count(1) as n_obs,
		sum(&target_var=1) as n_bads,
		sum(&target_var=0) as n_goods,
		calculated n_bads / n_obs as bad_rate format percent8.2
	from abt.train
	union all
	select
		'Validation' as dataset,
		count(1) as n_obs,
		sum(&target_var=1) as n_bads,
		sum(&target_var=0) as n_goods,
		calculated n_bads / n_obs as bad_rate format percent8.2
	from abt.valid;
quit;

proc freq data=abt.train;
	tables &target_var / nocum;
	title "Train set - Target distribution";
run;

proc freq data=abt.valid;
	tables &target_var / nocum;
	title "Validation set - Target distribution";
run;

proc means data=abt.train n;
	title "Train set size";
run;

proc means data=abt.valid n;
	title "Validation set size";
run;