-- What is the total spend per country?
-- What is the total spend per product category per country?
-- Which products are the most popular in each country?

SELECT 
	country,
	SUM(amtliq) AS spend_alcohol,
	SUM(amtvege) AS spend_vegetables,
	SUM(amtnonveg) AS spend_meat,
	SUM(amtpes) AS spend_fish,
	SUM(amtchocolates) AS spend_chocolates,
	SUM(amtcomm) AS spend_commodities,
	SUM(amtliq + amtvege + amtnonveg + amtpes + amtchocolates +amtcomm) AS total_spend,
	GREATEST(
	SUM(amtliq),
	SUM(amtvege),
	SUM(amtnonveg),
	SUM(amtpes),
	SUM(amtchocolates),
	SUM(amtcomm) 
	) AS highest_category_total
FROM
	marketing_data
GROUP BY 
	country
ORDER BY 
	highest_category_total DESC;

*/

-- Which products are the most popular based on marital status?

SELECT 
	marital_status,
	SUM(amtliq) AS spend_alcohol,
	SUM(amtvege) AS spend_vegetables,
	SUM(amtnonveg) AS spend_meat,
	SUM(amtpes) AS spend_fish,
	SUM(amtchocolates) AS spend_chocolates,
	SUM(amtcomm) AS spend_commodities,
	SUM(amtliq + amtvege + amtnonveg + amtpes + amtchocolates +amtcomm) AS total_spend,
	GREATEST(
	SUM(amtliq),
	SUM(amtvege),
	SUM(amtnonveg),
	SUM(amtpes),
	SUM(amtchocolates),
	SUM(amtcomm) 
	) AS highest_category_total
FROM
	marketing_data
GROUP BY 
	marital_status
ORDER BY 
	highest_category_total DESC;

*/

-- Which products are the most popular based on whether or not there are teens in the home?

SELECT
	CASE
		WHEN teenhome=0 THEN 'no teens'
		WHEN teenhome=1 THEN 'one teen'
		ELSE 'two teenagers'
		END AS teens,
	SUM(amtliq) AS spend_alcohol,
	SUM(amtvege) AS spend_vegetables,
	SUM(amtnonveg) AS spend_meat,
	SUM(amtpes) AS spend_fish,
	SUM(amtchocolates) AS spend_chocolates,
	SUM(amtcomm) AS spend_commodities,
	SUM(amtliq + amtvege + amtnonveg + amtpes + amtchocolates +amtcomm) AS total_spend
FROM 
	marketing_data
GROUP BY
	teens
ORDER BY
	total_spend DESC;

*/

-- Which products are the most popular based on whether or not there are kids in the home?

SELECT
	CASE
		WHEN kidhome=0 THEN 'no kids'
		WHEN kidhome=1 THEN 'one kid'
		ELSE 'two kids'
		END AS kids,
	SUM(amtliq) AS spend_alcohol,
	SUM(amtvege) AS spend_vegetables,
	SUM(amtnonveg) AS spend_meat,
	SUM(amtpes) AS spend_fish,
	SUM(amtchocolates) AS spend_chocolates,
	SUM(amtcomm) AS spend_commodities,
	SUM(amtliq + amtvege + amtnonveg + amtpes + amtchocolates +amtcomm) AS total_spend
FROM 
	marketing_data
GROUP BY
	kids
ORDER BY
	total_spend DESC;	

*/


