
-- Which is the most effective method of advertising in each country? 
-- (In this case, consider the total number of lead conversions as a measure of effectiveness).

SELECT 
	country,
	SUM(bulkmail_ad) AS bulkmail, 
	SUM(twitter_ad) AS twitter, 
	SUM(instagram_ad) AS instagram, 
	SUM(facebook_ad) AS facebook, 
	SUM(brochure_ad) AS brochure,
	SUM(bulkmail_ad+brochure_ad) AS total_printed_lead_conversions,
	SUM(twitter_ad+instagram_ad+facebook_ad) AS total_digital_lead_conversion,
	SUM(bulkmail_ad+twitter_ad+instagram_ad+facebook_ad+ brochure_ad) AS total_lead_conversions
FROM 
	marketing_data
JOIN 
	ad_data
ON 
	marketing_data.customer_id = ad_data.customer_id
GROUP BY 
	country
ORDER BY 
	total_lead_conversions DESC;

-- Which is the most effective method of advertising based on marital status?
-- (In this case, consider the total number of lead conversions as a measure of effectiveness).

SELECT 
	marital_status,
	SUM(bulkmail_ad) AS bulkmail, 
	SUM(twitter_ad) AS twitter, 
	SUM(instagram_ad) AS instagram, 
	SUM(facebook_ad) AS facebook, 
	SUM(brochure_ad) AS brochure,
	SUM(bulkmail_ad+brochure_ad) AS total_printed_lead_conversions,
	SUM(twitter_ad+instagram_ad+facebook_ad) AS total_digital_lead_conversion,
	SUM(bulkmail_ad+twitter_ad+instagram_ad+facebook_ad+ brochure_ad) AS total_lead_conversions
FROM 
	marketing_data
JOIN 
	ad_data
ON 
	marketing_data.customer_id = ad_data.customer_id
GROUP BY 
	marital_status
ORDER BY 
	total_lead_conversions DESC;

-- Assignment Activity 5(b)(iii)
-- Which social media platform(s) seem to be the most effective per country? 
-- (In this case, assume that purchases were in some way influenced by lead conversions from a campaign).

SELECT 
	m.country,m.response,
	SUM(a.twitter_ad) AS twitter, 
	SUM(a.instagram_ad) AS instagram, 
	SUM(a.facebook_ad) AS facebook,
	SUM(m.amtliq) AS alcohol_spend,
	SUM(m.amtvege) AS veg_spend,
	SUM(m.amtnonveg) AS non_veg_spend,
	SUM(m.amtpes) AS fish_spend,
	SUM(m.amtchocolates) AS chocolates_spend,
	SUM(m.amtcomm) AS commodity_spend,
	SUM(m.amtliq+m.amtvege+m.amtnonveg+m.amtpes+m.amtchocolates+m.amtcomm+a.twitter_ad+a.instagram_ad+a.facebook_ad) AS total_sales 
FROM
	marketing_data m
JOIN 
	ad_data a
ON 
	m.customer_id = a.customer_id
WHERE 
	response = True
GROUP BY 
	country, response
ORDER BY total_sales DESC;



