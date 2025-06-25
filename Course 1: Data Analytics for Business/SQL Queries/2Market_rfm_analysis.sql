-- RFM analysis

-- Which country has the highest average RFM score?

SELECT 
    m.country, 
    ROUND(AVG(r.recency_score), 2) AS avg_recency_score,
    ROUND(AVG(r.frequency_score), 2) AS avg_frequency_score,
    ROUND(AVG(r.monetary_score), 2) AS avg_monetary_score,
    ROUND(AVG(r.rfm_score), 2) AS avg_rfm_score
FROM 
    marketing_data m
JOIN 
    rfm_score r ON m.customer_id = r.customer_id
GROUP BY 
    m.country
ORDER BY
	avg_rfm_score DESC;

-- Which marital_status has the highest average RFM score?

SELECT 
    m.marital_status, 
    ROUND(AVG(r.recency_score), 2) AS avg_recency_score,
    ROUND(AVG(r.frequency_score), 2) AS avg_frequency_score,
    ROUND(AVG(r.monetary_score), 2) AS avg_monetary_score,
    ROUND(AVG(r.rfm_score), 2) AS avg_rfm_score
FROM 
    marketing_data m
JOIN 
    rfm_score r ON m.customer_id = r.customer_id
GROUP BY 
    m.marital_status
ORDER BY
	avg_rfm_score DESC;

-- Which level of education has the highest average RFM score?

SELECT 
    m.education, 
    ROUND(AVG(r.recency_score), 2) AS avg_recency_score,
    ROUND(AVG(r.frequency_score), 2) AS avg_frequency_score,
    ROUND(AVG(r.monetary_score), 2) AS avg_monetary_score,
    ROUND(AVG(r.rfm_score), 2) AS avg_rfm_score
FROM 
    marketing_data m
JOIN 
    rfm_score r ON m.customer_id = r.customer_id
GROUP BY 
    m.education
ORDER BY
	avg_rfm_score DESC;

-- Average RFM score for customers with teens at home.

SELECT 
    m.teenhome,
    ROUND(AVG(r.recency_score), 2) AS avg_recency_score,
    ROUND(AVG(r.frequency_score), 2) AS avg_frequency_score,
    ROUND(AVG(r.monetary_score), 2) AS avg_monetary_score,
    ROUND(AVG(r.rfm_score), 2) AS avg_rfm_score
FROM 
    marketing_data m
JOIN 
    rfm_score r ON m.customer_id = r.customer_id
GROUP BY 
    m.teenhome
ORDER BY
	avg_rfm_score DESC;


-- Average RFM score for customers with kids at home.

SELECT 
    m.kidhome,
    ROUND(AVG(r.recency_score), 2) AS avg_recency_score,
    ROUND(AVG(r.frequency_score), 2) AS avg_frequency_score,
    ROUND(AVG(r.monetary_score), 2) AS avg_monetary_score,
    ROUND(AVG(r.rfm_score), 2) AS avg_rfm_score
FROM 
    marketing_data m
JOIN 
    rfm_score r ON m.customer_id = r.customer_id
GROUP BY 
    m.kidhome
ORDER BY
	avg_rfm_score DESC;

-- Who are the customers with an RFM score of 4?

SELECT 
    m.customer_id,
	m.age, 
	m.income, 
	m.marital_status, 
	m.education, 
	m.country, 
	m.teenhome, 
	m.kidhome,
    r.rfm_score
FROM 
    marketing_data m
JOIN 
    rfm_score r ON m.customer_id = r.customer_id
GROUP BY
    m.customer_id, r.rfm_score
HAVING 
    rfm_score = 4;

*/

-- What is the average age of the customers with an RFM score of 4?

SELECT 
	ROUND(AVG(m.age),2) AS avg_age_of_high_rfm_customers
FROM 
    marketing_data m
JOIN 
    rfm_score r ON m.customer_id = r.customer_id
WHERE
    rfm_score = 4;

-- What is the average income of the customers with an RFM score of 4?

SELECT 
	ROUND(AVG(m.income),2) AS avg_age_of_high_rfm_customers
FROM 
    marketing_data m
JOIN 
    rfm_score r ON m.customer_id = r.customer_id
WHERE
    rfm_score = 4;

-- Average rfm score for customers with succesful lead conversion via brochure_ad

SELECT 
    ROUND(AVG(r.rfm_score), 2) AS avg_rfm
FROM 
    rfm_score r
JOIN 
    ad_data a 
ON 
	r.customer_id = a.customer_id
WHERE 
	brochure_ad = 1;


-- Average RFM score by every country based on lead conversion success/failure per ad.

SELECT 
    m.country, a.brochure_ad,
    ROUND(AVG(r.rfm_score), 2) AS avg_rfm_score
FROM 
    marketing_data m
JOIN 
    rfm_score r ON m.customer_id = r.customer_id
JOIN
	ad_data a ON m.customer_id = a.customer_id
WHERE 
	brochure_ad = 1
GROUP BY 
    m.country, a.brochure_ad
ORDER BY
	avg_rfm_score DESC;
