# SQL Cheaet sheet



## **SQL**(MySQL or Oracle SQL)



### Rank Functions

* `ROW_NUMBER()`: for same score, assgin different ranks

  ```sql
  SELECT Studentname, 
         Subject, 
         Marks, 
         ROW_NUMBER() OVER(ORDER BY Marks) RowNumber
  FROM ExamResult;
  ```

  ![](https://www.sqlshack.com/wp-content/uploads/2019/07/row_number-sql-rank-function.png)

* `RANK()`:

  ```sql
  SELECT Studentname, 
         Subject, 
         Marks, 
         RANK() OVER(PARTITION BY Studentname ORDER BY Marks DESC) Rank
  FROM ExamResult
  ORDER BY Studentname, 
           Rank;
  ```

  

  ![](https://www.sqlshack.com/wp-content/uploads/2019/07/ranksql-rank-function.png)

* `DENSE_RANK()`: if we have duplicate values, SQL assigns different ranks to those rows as well. Ideally, we should get the same rank for duplicate or similar values.

  ```sql
  SELECT Studentname, 
         Subject, 
         Marks, 
         DENSE_RANK() OVER(ORDER BY Marks DESC) Rank
  FROM ExamResult
  ORDER BY Rank;
  ```

  

  ![](https://www.sqlshack.com/wp-content/uploads/2019/07/dense_ranksql-rank-function.png)

### Sum with partition

```cmd
Scores table:
+-------------+--------+------------+--------------+
| player_name | gender | day        | score_points |
+-------------+--------+------------+--------------+
| Aron        | F      | 2020-01-01 | 17           |
| Alice       | F      | 2020-01-07 | 23           |
| Bajrang     | M      | 2020-01-07 | 7            |
| Khali       | M      | 2019-12-25 | 11           |
| Slaman      | M      | 2019-12-30 | 13           |
| Joe         | M      | 2019-12-31 | 3            |
| Jose        | M      | 2019-12-18 | 2            |
| Priya       | F      | 2019-12-31 | 23           |
| Priyanka    | F      | 2019-12-30 | 17           |
+-------------+--------+------------+--------------+
Result table:
+--------+------------+-------+
| gender | day        | total |
+--------+------------+-------+
| F      | 2019-12-30 | 17    |
| F      | 2019-12-31 | 40    |
| F      | 2020-01-01 | 57    |
| F      | 2020-01-07 | 80    |
| M      | 2019-12-18 | 2     |
| M      | 2019-12-25 | 13    |
| M      | 2019-12-30 | 26    |
| M      | 2019-12-31 | 29    |
| M      | 2020-01-07 | 36    |
+--------+------------+-------+
```

```sql
-- Write an SQL query to find the total score for each gender at each day.
select gender, day, sum(score_points) over (partition by gender order by day) as total
from Scores
order by gender, day;
```



### Consecutive Days

```sql
--find id with more than 5 consecutive days
SELECT 
	DISTINCT x.id
FROM 
	Demo_table x, Demo_table y
WHERE 
	x.id=y.id AND
	DATEDIFF(x.demo_date, y.demo_date) BETWEEN 1 AND 4
GROUP BY 
	x.id, x.demo_date
HAVING 
	COUNT(DISTINCT y.demo_date)=4;

--with lag() function
select 
	id,  
	demo_date, 
	lag(demo_date,5,'1990-01-01') over ( partition by l.id order by demo_date) as lag_5_days 
from 
	Demo_table 
```

```sql
--with dense_rank
SELECT  id,
        demo_date,
        DATE_ADD(demo_date,INTERVAL
        	-(dense_rank() OVER(PARTITION BY id ORDER BY demo_date)-1) DAY) GroupingSet
    FROM Demo_table
```



### Find item having at least value

```sql
select item form  table
group by item
having count(*) >= num;
```



### Calculate moving average with `datediff`

```cmd
Customer table:
+-------------+--------------+--------------+-------------+
| customer_id | name         | visited_on   | amount      |
+-------------+--------------+--------------+-------------+
| 1           | Jhon         | 2019-01-01   | 100         |
| 2           | Daniel       | 2019-01-02   | 110         |
| 3           | Jade         | 2019-01-03   | 120         |
| 4           | Khaled       | 2019-01-04   | 130         |
| 5           | Winston      | 2019-01-05   | 110         | 
| 6           | Elvis        | 2019-01-06   | 140         | 
| 7           | Anna         | 2019-01-07   | 150         |
| 8           | Maria        | 2019-01-08   | 80          |
| 9           | Jaze         | 2019-01-09   | 110         | 
| 1           | Jhon         | 2019-01-10   | 130         | 
| 3           | Jade         | 2019-01-10   | 150         | 
+-------------+--------------+--------------+-------------+

Result table:
+--------------+--------------+----------------+
| visited_on   | amount       | average_amount |
+--------------+--------------+----------------+
| 2019-01-07   | 860          | 122.86         |
| 2019-01-08   | 840          | 120            |
| 2019-01-09   | 840          | 120            |
| 2019-01-10   | 1000         | 142.86         |
+--------------+--------------+----------------+
```



```sql
select
    a.visited_on,
    sum(b.tot) as amount,
    round(avg(b.tot),2) as average_amount
from
    (select distinct visited_on from Customer) a
    left join
    (select visited_on, sum(amount) as tot
    from Customer
    group by visited_on) b
on
    datediff(a.visited_on, b.visited_on) <= 6
    and
    a.visited_on >= b.visited_on
group by
    a.visited_on
having
    count(a.visited_on) = 7
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
-- MySQL
-- Convert the current date and time to string (year-month-day)
-- SELECT DATE_FORMAT(SYSDATE(), '%Y-%m-%d');
group by Movies.title
order by avg_rating desc, Movies.title asc
)
where rownum = 1
```



### Use concat to generate text

```sql
SELECT "There are a total of", COUNT(OCCUPATION), concat(LOWER(OCCUPATION),"s.") 
FROM OCCUPATIONS GROUP BY OCCUPATION ORDER BY COUNT(OCCUPATION) ASC, OCCUPATION ASC;
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


### Index

Indexes are **special lookup tables** that the database search engine can use to speed up data retrieval.An index helps to speed up **SELECT** queries and **WHERE** clauses, but it slows down data input, with the **UPDATE** and the **INSERT** statements. Indexes can be created or dropped with no effect on the data. 
Creating an index involves the **CREATE INDEX** statement, which allows you to name the index, to specify the table and which column or columns to index, and to indicate whether the index is in an ascending or descending order.
Indexes can also be unique, like the **UNIQUE** constraint, in that the index prevents duplicate entries in the column or combination of columns on which there is an index.

#### CREATE INDEX Syntax

Creates an index on a table. Duplicate values are allowed:

```sql
CREATE INDEX index_name ON table_name;
```

```sql
CREATE INDEX index_name
ON table_name (column1, column2, ...);
```

#### CREATE UNIQUE INDEX Syntax

Creates a unique index on a table. Duplicate values are not allowed:

```sql
CREATE UNIQUE INDEX index_name
ON table_name (column1, column2, ...);
```
#### DROP INDEX Statement

The DROP INDEX statement is used to delete an index in a table.

```sql
--SQL Server
DROP INDEX table_name.index_name;
--Oracle
DROP INDEX index_name;
--MySQL
ALTER TABLE table_name
DROP INDEX index_name;
```

---



## PL/SQL

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

