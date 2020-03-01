# <center>SQL Cheaet sheet</center>

## **SQL**(MySQL or  Oracle SQL)

### Find item having at least value

```sql
select item form  table
group by item
having count(*) >= num;
```

### Convert date time

```sql
SELECT TO_DATE('2012-06-05', 'YYYY-MM-DD') FROM dual;
```

### Find date period

```sql

#MySQL
WHERE first_login >= DATE_ADD('2019-06-30', INTERVAL -90 DAY)
#Oracle SQL
where first_date >= (to_date( '2019-06-30','YYYY-MM-DD')-90) 
```

### Find record in certain month

> Can't use rownum directly after a group by process, have to select the whole result set then return the first 1 row.

```sql
select title from
(
select Movies.title ,avg(Movie_Rating.rating) as avg_rating
from Movie_Rating
left join Movies
on Movies.movie_id = Movie_Rating.movie_id
where to_char(Movie_Rating.created_at, 'mm') = 02
group by Movies.title
order by avg_rating desc, Movies.title asc
)
where rownum = 1
```

### Find item looks like some value

```sql
SELECT column1, column2, ...
FROM table_name
WHERE columnN LIKE pattern;
```

|LIKE Operator|Description|
|--------------|-----------|
|WHERE CustomerName LIKE 'a%'|Finds any values that start with "a"|
|WHERE CustomerName LIKE '%a'|Finds any values that end with "a"|
|WHERE CustomerName LIKE '%or%'|Finds any values that have "or" in any position|
|WHERE CustomerName LIKE '_r%'|Finds any values that have "r" in the second position|
|WHERE CustomerName LIKE 'a__%'|Finds any values that start with "a" and are at least 3 characters in length|
|WHERE ContactName LIKE 'a%o'|Finds any values that start with "a" and ends with "o"|

### With Statement

```sql
with CTE as
(select S.product_id, P.product_name, S.buyer_id from Sales S left join Product P on S.product_id = P.product_id)

select distinct buyer_id from CTE where buyer_id not in (select buyer_id from CTE where product_name = 'iPhone') and product_name = 'S8'
```

### Group by and show count with zero value

```sql
SELECT posts.sub_id AS post_id,
       nvl(count(DISTINCT comments.sub_id), 0) AS number_of_comments
FROM Submissions posts
LEFT JOIN Submissions comments ON posts.sub_id = comments.parent_id
WHERE posts.parent_id IS NULL
GROUP BY posts.sub_id
ORDER BY posts.sub_id;
```

## **PL/SQL**

### Output in Oracle/SQL

```sql
dbms_output.put_line('student name is  ' || v_variable_name);
```

### Find Median number

```sql
SUM(CASE
    WHEN Employee.Salary = alias.Salary THEN 1
    ELSE 0
END) >= ABS(SUM(SIGN(Employee.Salary - alias.Salary)))
# sign() return the sign of the number: if positive then 1, negative then -1
```

>The median's frequency should be equal or grater than the absolute difference of its bigger elements and small ones in an array no matter whether it has odd or even amount of numbers and whether they are distinct. This rule is the key, and it is represented as the previous code.

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
--example 1
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

--example 2
SELECT CustomerName, City, Country
FROM Customers
ORDER BY
(CASE
    WHEN City IS NULL THEN Country
    ELSE City
END);

```

### Create table if it does not exist

```sql
DECLARE

    temp_nm        INT;
BEGIN
    SELECT
        COUNT(*)
    INTO temp_nm
    FROM
        user_tables
    WHERE
        table_name = upper('hw6_employee_table');

    IF temp_nm = 0 THEN
--create the table if it not exist
EXECUTE IMMEDIATE 'CREATE TABLE hw6_employee_table
      ( EMPLOYEE_ID number(6),
       FIRST_NAME varchar2(255),
       LAST_NAME varchar2(255),
       EMAIL varchar2(255),
       PHONE_NUMBER varchar2(255),
       HIRE_DATE date,
       JOB_ID varchar2(255),
       SALARY number(6),
       COMMISSION_PCT number(6),
       MANAGER_ID number(6),
       DEPARTMENT_ID number(6)
      )'
        ;
END IF;
end;
```

### Triggers

```sql
create or replace trigger trig_cl12_population_pk
    before insert on cl12_population
    for each row
declare
    v_population_pk cl12_population.population_pk%type;
begin
    if :new.population_pk is null then
    select seq_cl12_population.nextval
    into v_population_pk
    from dual;
    
    :new.population_pk := v_population_pk;
    end if;
end;



```
---

### [Markdown Demo](https://markdown-it.github.io/)