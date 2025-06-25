create or replace view tableau.customer_rfm_4 as
SELECT 
    m.customer_id,
	m.age, 
	m.income, 
	m.marital_status, 
	m.education, 
	m.country, 
	m.teenhome, 
	m.kidhome,
	m.amtliq,
	m.amtvege,
	m.amtnonveg,
	m.amtpes,
	m.amtchocolates,
	m.amtcomm,
	m.response,
	m.count_success,
    r.rfm_score
FROM 
    marketing_data m
JOIN 
    rfm_score r ON m.customer_id = r.customer_id
GROUP BY
    m.customer_id, r.rfm_score
HAVING 
    rfm_score = 4;