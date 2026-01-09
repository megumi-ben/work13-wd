#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"

#include "mb/pg_wchar.h"
#include "regex/regex.h"

#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/errcodes.h"
#include "catalog/pg_type.h"

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(regex_match_span);
PG_FUNCTION_INFO_V1(regex_find_from);
PG_FUNCTION_INFO_V1(regex_find_all);
PG_FUNCTION_INFO_V1(regex_match_from_upto);
PG_FUNCTION_INFO_V1(regex_find_at_upto);
PG_FUNCTION_INFO_V1(regex_verify_windows);

/* Helper to convert text to pg_wchar buffer (null-terminated) and return length in characters */
static int
text_to_wchar(text *src, pg_wchar **dest)
{
	int			len_bytes = VARSIZE_ANY_EXHDR(src);
	const char *data = VARDATA_ANY(src);
	pg_wchar   *wstr;
	int			wlen;

	/* wlen <= len_bytes, so (len_bytes + 1) is safe */
	wstr = (pg_wchar *) palloc((len_bytes + 1) * sizeof(pg_wchar));
	wlen = pg_mb2wchar_with_len(data, wstr, len_bytes);
	wstr[wlen] = (pg_wchar) 0;
	*dest = wstr;
	return wlen;
}

static regex_t *
compile_regex(text *pattern, Oid collation)
{
	int cflags = REG_ADVANCED;
	return RE_compile_and_cache(pattern, cflags, collation);
}

/*
 * IMPORTANT: do NOT name the argument "errcode" (it conflicts with errcode(...) macro)
 */
static void
report_regerror(int re_rc, regex_t *re)
{
	char errMsg[1024];

	pg_regerror(re_rc, re, errMsg, sizeof(errMsg));
	ereport(ERROR,
			(errcode(ERRCODE_INVALID_REGULAR_EXPRESSION),
			 errmsg("regular expression failed: %s", errMsg)));
}

Datum
regex_match_span(PG_FUNCTION_ARGS)
{
	text	   *txt = PG_GETARG_TEXT_PP(0);
	text	   *pattern = PG_GETARG_TEXT_PP(1);
	int32		start_pos = PG_GETARG_INT32(2);
	int32		end_pos = PG_GETARG_INT32(3);

	pg_wchar   *data;
	int			wide_len;

	regex_t    *re;
	regmatch_t	pmatch[1];
	int			regexec_result;

	Oid			collation = PG_GET_COLLATION();
	int			search_start;
	size_t		wide_len_sz;
	size_t		search_start_sz;

	wide_len = text_to_wchar(txt, &data);

	if (start_pos < 1 || end_pos < start_pos || start_pos > wide_len || end_pos > wide_len)
		PG_RETURN_BOOL(false);

	search_start = start_pos - 1;
	wide_len_sz = (size_t) wide_len;
	search_start_sz = (size_t) search_start;

	re = compile_regex(pattern, collation);

	regexec_result = pg_regexec(re,
								data,
								wide_len_sz,
								search_start_sz,
								NULL,
								1,
								pmatch,
								0);

	if (regexec_result == REG_NOMATCH)
		PG_RETURN_BOOL(false);
	else if (regexec_result != REG_OKAY)
		report_regerror(regexec_result, re);

	/* rm_so is 0-based; rm_eo behaves like an end position compatible with your checks */
	if (pmatch[0].rm_so == search_start && pmatch[0].rm_eo == end_pos)
		PG_RETURN_BOOL(true);

	PG_RETURN_BOOL(false);
}

Datum
regex_match_from_upto(PG_FUNCTION_ARGS)
{
	text	   *txt = PG_GETARG_TEXT_PP(0);
	text	   *pattern = PG_GETARG_TEXT_PP(1);
	int32		start_pos = PG_GETARG_INT32(2);
	int32		end_pos = PG_GETARG_INT32(3);

	pg_wchar   *data;
	int			wide_len;

	regex_t    *re;
	regmatch_t	pmatch[1];
	int			regexec_result;

	Oid			collation = PG_GET_COLLATION();
	int			search_start;

	wide_len = text_to_wchar(txt, &data);

	if (start_pos < 1 || end_pos < start_pos || start_pos > wide_len || end_pos > wide_len)
		PG_RETURN_BOOL(false);

	search_start = start_pos - 1;
	re = compile_regex(pattern, collation);

	regexec_result = pg_regexec(re,
								data,
								(size_t) wide_len,
								(size_t) search_start,
								NULL,
								1,
								pmatch,
								0);

	if (regexec_result == REG_NOMATCH)
		PG_RETURN_BOOL(false);
	else if (regexec_result != REG_OKAY)
		report_regerror(regexec_result, re);

	if (pmatch[0].rm_so == search_start && pmatch[0].rm_eo <= end_pos)
		PG_RETURN_BOOL(true);

	PG_RETURN_BOOL(false);
}

Datum
regex_find_at_upto(PG_FUNCTION_ARGS)
{
	text	   *txt = PG_GETARG_TEXT_PP(0);
	text	   *pattern = PG_GETARG_TEXT_PP(1);
	int32		start_pos = PG_GETARG_INT32(2);
	int32		end_pos = PG_GETARG_INT32(3);
	bool		allow_empty = (PG_NARGS() > 4) ? PG_GETARG_BOOL(4) : false;

	pg_wchar   *data;
	int			wide_len;

	regex_t    *re;
	regmatch_t	pmatch[1];
	int			regexec_result;

	Oid			collation = PG_GET_COLLATION();
	int			search_start;

	wide_len = text_to_wchar(txt, &data);

	if (start_pos < 1 || end_pos < start_pos || start_pos > wide_len || end_pos > wide_len)
		PG_RETURN_NULL();

	search_start = start_pos - 1;
	re = compile_regex(pattern, collation);

	regexec_result = pg_regexec(re,
								data,
								(size_t) wide_len,
								(size_t) search_start,
								NULL,
								1,
								pmatch,
								0);

	if (regexec_result == REG_NOMATCH)
		PG_RETURN_NULL();
	else if (regexec_result != REG_OKAY)
		report_regerror(regexec_result, re);

	if (pmatch[0].rm_so != search_start || pmatch[0].rm_eo > end_pos)
		PG_RETURN_NULL();

	if (!allow_empty && pmatch[0].rm_so == pmatch[0].rm_eo)
		PG_RETURN_NULL();

	{
		Datum		elements[2];
		ArrayType  *result;

		/* return {start(1-based), end} per your current convention */
		elements[0] = Int32GetDatum((int32) (pmatch[0].rm_so + 1));
		elements[1] = Int32GetDatum((int32) pmatch[0].rm_eo);

		result = construct_array(elements, 2, INT4OID, sizeof(int32), true, 'i');
		PG_RETURN_ARRAYTYPE_P(result);
	}
}

Datum
regex_find_from(PG_FUNCTION_ARGS)
{
	text	   *txt = PG_GETARG_TEXT_PP(0);
	text	   *pattern = PG_GETARG_TEXT_PP(1);
	int32		start_pos = PG_GETARG_INT32(2);
	bool		allow_empty = (PG_NARGS() > 3) ? PG_GETARG_BOOL(3) : false;

	pg_wchar   *data;
	int			wide_len;

	regex_t    *re;
	regmatch_t	pmatch[1];
	int			regexec_result;

	Oid			collation = PG_GET_COLLATION();
	int			search_start;
	size_t		wide_len_sz;
	size_t		search_start_sz;

	wide_len = text_to_wchar(txt, &data);

	/* Allow searching at position wide_len+1 to permit end-anchored zero-length matches */
	if (start_pos < 1 || start_pos > wide_len + 1)
		PG_RETURN_NULL();

	search_start = start_pos - 1;
	wide_len_sz = (size_t) wide_len;

	re = compile_regex(pattern, collation);

	while (search_start <= wide_len)
	{
		search_start_sz = (size_t) search_start;

		regexec_result = pg_regexec(re,
									data,
									wide_len_sz,
									search_start_sz,
									NULL,
									1,
									pmatch,
									0);

		if (regexec_result == REG_NOMATCH)
			PG_RETURN_NULL();
		else if (regexec_result != REG_OKAY)
			report_regerror(regexec_result, re);

		if (!allow_empty && pmatch[0].rm_so == pmatch[0].rm_eo)
		{
			search_start = pmatch[0].rm_eo + 1;
			continue;
		}

		{
			Datum		elements[2];
			ArrayType  *result;

			elements[0] = Int32GetDatum((int32) (pmatch[0].rm_so + 1));
			elements[1] = Int32GetDatum((int32) pmatch[0].rm_eo);

			result = construct_array(elements, 2, INT4OID, sizeof(int32), true, 'i');
			PG_RETURN_ARRAYTYPE_P(result);
		}
	}

	PG_RETURN_NULL();
}

static int
verify_window_hits(regex_t *re, pg_wchar *data, int wide_len,
				   int32 *windows, int nwindows, bool allow_empty)
{
	int i;

	for (i = 0; i < nwindows; i++)
	{
		int32		start_pos = windows[2 * i];
		int32		end_pos = windows[2 * i + 1];
		int			search_start;

		regmatch_t	pmatch[1];
		int			regexec_result;

		if (start_pos < 1 || end_pos < start_pos || start_pos > wide_len || end_pos > wide_len)
			continue;

		search_start = start_pos - 1;

		regexec_result = pg_regexec(re,
									data,
									(size_t) wide_len,
									(size_t) search_start,
									NULL,
									1,
									pmatch,
									0);

		if (regexec_result == REG_NOMATCH)
			continue;
		else if (regexec_result != REG_OKAY)
			report_regerror(regexec_result, re);

		if (pmatch[0].rm_so != search_start)
			continue;
		if (!allow_empty && pmatch[0].rm_so == pmatch[0].rm_eo)
			continue;
		if (pmatch[0].rm_eo <= end_pos)
			return i;
	}

	return -1;
}

Datum
regex_verify_windows(PG_FUNCTION_ARGS)
{
	text	   *txt = PG_GETARG_TEXT_PP(0);
	text	   *pattern = PG_GETARG_TEXT_PP(1);
	ArrayType  *windows_array = PG_GETARG_ARRAYTYPE_P(2);
	bool		allow_empty = (PG_NARGS() > 3) ? PG_GETARG_BOOL(3) : false;

	int32	   *windows;
	int			nelems;
	int			nwindows;

	pg_wchar   *data;
	int			wide_len;
	regex_t    *re;
	int			hit;

	if (ARR_NDIM(windows_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("windows must be a 1-D int4 array")));

	if (ARR_HASNULL(windows_array))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("windows array must not contain nulls")));

	nelems = ArrayGetNItems(ARR_NDIM(windows_array), ARR_DIMS(windows_array));
	if (nelems == 0)
		PG_RETURN_INT32(-1);
	if (nelems % 2 != 0)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("windows array length must be even (start/end pairs)")));

	windows = (int32 *) ARR_DATA_PTR(windows_array);
	nwindows = nelems / 2;

	wide_len = text_to_wchar(txt, &data);
	re = compile_regex(pattern, PG_GET_COLLATION());

	hit = verify_window_hits(re, data, wide_len, windows, nwindows, allow_empty);

	PG_RETURN_INT32(hit);
}

typedef struct
{
	int		   *match_locs;		/* start/end pairs (1-based start, end per current convention) */
	int			nmatches;		/* number of matches recorded */
	int			next_idx;		/* next match index to return */
	bool		overlap;		/* whether to allow overlapping matches */
} regex_findall_ctx;

Datum
regex_find_all(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	regex_findall_ctx *ctx;

	if (SRF_IS_FIRSTCALL())
	{
		text	   *txt;
		text	   *pattern;
		int32		start_pos;
		bool		overlap;
		bool		allow_empty;
		Oid			collation;

		pg_wchar   *data;
		int			wide_len;
		size_t		wide_len_sz;

		regex_t    *re;
		int			start_search;

		int			array_len;
		int			array_idx;

		MemoryContext oldcontext;

		txt = PG_GETARG_TEXT_PP(0);
		pattern = PG_GETARG_TEXT_PP(1);
		start_pos = PG_GETARG_INT32(2);
		overlap = (PG_NARGS() > 3) ? PG_GETARG_BOOL(3) : true;
		allow_empty = (PG_NARGS() > 4) ? PG_GETARG_BOOL(4) : false;
		collation = PG_GET_COLLATION();

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		wide_len = text_to_wchar(txt, &data);
		wide_len_sz = (size_t) wide_len;

		ctx = (regex_findall_ctx *) palloc(sizeof(regex_findall_ctx));
		ctx->match_locs = NULL;
		ctx->nmatches = 0;
		ctx->next_idx = 0;
		ctx->overlap = overlap;

		if (start_pos >= 1 && start_pos <= wide_len + 1)
		{
			re = compile_regex(pattern, collation);

			start_search = start_pos - 1;

			array_len = 32; /* number of pairs capacity */
			ctx->match_locs = (int *) palloc(sizeof(int) * array_len * 2);
			array_idx = 0;

			for (;;)
			{
				regmatch_t	pmatch[1];
				int			regexec_result;

				if (start_search > wide_len)
					break;

				regexec_result = pg_regexec(re,
											data,
											wide_len_sz,
											(size_t) start_search,
											NULL,
											1,
											pmatch,
											0);

				if (regexec_result == REG_NOMATCH)
					break;
				else if (regexec_result != REG_OKAY)
					report_regerror(regexec_result, re);

				if (!allow_empty && pmatch[0].rm_so == pmatch[0].rm_eo)
				{
					start_search = pmatch[0].rm_eo + 1;
					continue;
				}

				if ((array_idx + 2) > array_len * 2)
				{
					array_len *= 2;
					ctx->match_locs = (int *) repalloc(ctx->match_locs, sizeof(int) * array_len * 2);
				}

				ctx->match_locs[array_idx++] = (int) (pmatch[0].rm_so + 1); /* 1-based start */
				ctx->match_locs[array_idx++] = (int) pmatch[0].rm_eo;       /* end per your convention */
				ctx->nmatches++;

				if (overlap)
					start_search = pmatch[0].rm_so + 1;
				else
					start_search = pmatch[0].rm_eo;

				if (pmatch[0].rm_so == pmatch[0].rm_eo)
					start_search++;
			}
		}

		funcctx->user_fctx = ctx;
		funcctx->max_calls = ctx->nmatches;

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	ctx = (regex_findall_ctx *) funcctx->user_fctx;

	if (ctx->next_idx >= ctx->nmatches)
		SRF_RETURN_DONE(funcctx);
	else
	{
		int			base = ctx->next_idx * 2;
		Datum		elements[2];
		ArrayType  *result;

		elements[0] = Int32GetDatum((int32) ctx->match_locs[base]);
		elements[1] = Int32GetDatum((int32) ctx->match_locs[base + 1]);
		ctx->next_idx++;

		result = construct_array(elements, 2, INT4OID, sizeof(int32), true, 'i');
		SRF_RETURN_NEXT(funcctx, PointerGetDatum(result));
	}
}
