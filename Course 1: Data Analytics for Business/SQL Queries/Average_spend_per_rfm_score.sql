SELECT 
    r.rfm_score,
    AVG(m.amtliq) AS avg_spend_alcohol,
    AVG(m.amtvege) AS avg_spend_vegetables,
    AVG(m.amtnonveg) AS avg_spend_meat,
    AVG(m.amtpes) AS avg_spend_fish,
    AVG(m.amtchocolates) AS avg_spend_chocolates,
    AVG(m.amtcomm) AS avg_spend_commodities,
    AVG(m.amtliq + m.amtvege + m.amtnonveg + m.amtpes + m.amtchocolates + m.amtcomm) AS avg_total_spend,
    GREATEST(
        AVG(m.amtliq),
        AVG(m.amtvege),
        AVG(m.amtnonveg),
        AVG(m.amtpes),
        AVG(m.amtchocolates),
        AVG(m.amtcomm) 
    ) AS highest_category_avg
FROM
    marketing_data m
JOIN  
    rfm_score r 
ON 
    m.customer_id = r.customer_id
GROUP BY 
    r.rfm_score;
