-- models/staging/stg_signalconso.sql
WITH source AS (
    SELECT * FROM {{ source('Complaints', 'Signal_Conso') }}
),

deduped AS (
    SELECT *
    FROM source
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY id
        ORDER BY creationdate DESC
    ) = 1
),

renamed AS (
    SELECT
        id                              AS source_id,
        creationdate                    AS created_at,

        category[SAFE_OFFSET(0)]        AS category,
        subcategories,
        tags,
        status,

        contactagreement,
        forwardtoreponseconso,
        signalement_transmis,
        signalement_lu,
        signalement_reponse,

        dep_name,
        dep_code,
        CAST(reg_code AS STRING)        AS reg_code,
        reg_name,

        clean_text,
        token_count,
        is_valid,

        ROW_NUMBER() OVER (
            ORDER BY creationdate DESC
        )                               AS _row_rank

    FROM deduped
)

SELECT * EXCEPT (_row_rank)
FROM renamed