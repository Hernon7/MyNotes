# SQL Cheaet sheet

## **SQL**(MySQL or  Oracle SQL)

### Find item having at least value

```sql
select item form  table
group by item
having count(*) >= num;
```

### Find item looks like some value

```sql
SELECT column1, column2, ...
FROM table_name
WHERE columnN LIKE pattern;
```

### Convert date time

```sql
SELECT TO_DATE('2012-06-05', 'YYYY-MM-DD') FROM dual;
```

|LIKE Operator|Description|
|--------------|-----------|
|WHERE CustomerName LIKE 'a%'|Finds any values that start with "a"|
|WHERE CustomerName LIKE '%a'|Finds any values that end with "a"|
|WHERE CustomerName LIKE '%or%'|Finds any values that have "or" in any position|
|WHERE CustomerName LIKE '_r%'|Finds any values that have "r" in the second position|
|WHERE CustomerName LIKE 'a__%'|Finds any values that start with "a" and are at least 3 characters in length|
|WHERE ContactName LIKE 'a%o'|Finds any values that start with "a" and ends with "o"|

## **PL/SQL**

### Output in Oracle/SQL

```sql
dbms_output.put_line('student name is  ' || v_variable_name);
```
### Loop

```sql
DECLARE
  sum INTEGER := 0;
BEGIN
  LOOP
    sum := sum + 1;
    IF sum > 10 THEN
       EXIT;
    END IF;
  END LOOP;
END  
```

### Case when

```sql
#example 1
DECLARE
   jobid      employees.job_id%TYPE;
   empid      employees.employee_id%TYPE := 115;
   sal_raise  NUMBER(3,2);
BEGIN
  SELECT job_id INTO jobid from employees WHERE employee_id = empid;
  CASE
    WHEN jobid = 'PU_CLERK' THEN sal_raise := .09;
    WHEN jobid = 'SH_CLERK' THEN sal_raise := .08;
    WHEN jobid = 'ST_CLERK' THEN sal_raise := .07;
    ELSE sal_raise := 0;
  END CASE;
END;
/

#example 2
SELECT CustomerName, City, Country
FROM Customers
ORDER BY
(CASE
    WHEN City IS NULL THEN Country
    ELSE City
END);

```

---

### [Markdown Demo](https://markdown-it.github.io/)