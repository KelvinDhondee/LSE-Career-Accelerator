-- Of those customers who have children, what is the breakdown of kids/teens based on marital status?

SELECT marital_status,
	SUM(kidhome) AS kids_at_home, 
	SUM(teenhome) AS teens_at_home
FROM marketing_data
GROUP BY marital_status;

--What is the profile for customers who have complained?

SELECT 
	customer_id, age, marital_status, income, teenhome, kidhome, recency, complain
FROM 
	marketing_data
WHERE
	complain='true'
GROUP BY
	marital_status, customer_id
ORDER BY
	income DESC;

-- What is the average recency days value of customers who have complained?

SELECT ROUND(AVG(recency)) AS recency_days_of_complainers
from (SELECT 
	customer_id, age, marital_status, income, teenhome, kidhome, recency, complain
FROM 
	marketing_data
WHERE
	complain='true'
GROUP BY
	marital_status, customer_id
ORDER BY
	income DESC);

-- What is the average recency days value of customers who have not complained?

SELECT ROUND(AVG(recency)) AS recency_days_of_complainers
from (SELECT 
	customer_id, age, marital_status, income, teenhome, kidhome, recency, complain
FROM 
	marketing_data
WHERE
	complain='false'
GROUP BY
	marital_status, customer_id
ORDER BY
	income DESC);


-- What is the profile for customers from ME?

SELECT *
FROM marketing_data
WHERE country = 'ME';

-- Total amount spent by ME customers for each product category.

SELECT 
	SUM(amtliq) AS me_liq, 
	SUM(amtvege) AS me_veg, 
	SUM(amtnonveg) AS me_nonveg, 
	SUM(amtpes) AS me_pes, 
	SUM(amtchocolates) AS me_choc, 
	SUM(amtcomm) AS me_comm
FROM marketing_data
WHERE country = 'ME';

*/
-- Average spend per country

SELECT country, SUM(amtliq+amtvege+amtnonveg+amtpes+amtchocolates+amtcomm)/COUNT(customer_id) AS average_spend
from marketing_data
GROUP BY country
ORDER BY average_spend DESC;