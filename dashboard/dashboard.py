import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
import geopandas as gpd
sns.set(style='dark')

pd.set_option('display.max_columns',None)

# memanggil data sebelumnya yang telah di merge
all_df = pd.read_csv("dashboard/all_data.csv")
datetime_columns = ['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date',
                    'order_delivered_customer_date','order_estimated_delivery_date','review_answer_timestamp',
                    'shipping_limit_date']

all_df.sort_values(by='order_purchase_timestamp', inplace=True)
all_df.reset_index(inplace=True)

for column in datetime_columns:
    all_df[column] = all_df[column].astype(str)
    all_df[column] = all_df[column].str.replace(r'\.\d+', '', regex=True)
    all_df[column] = pd.to_datetime(all_df[column])

# memanggil koordinat database brazil
brazil_states = gpd.read_file('geopandas-brasil/shapes/gadm36_BRA_1.shp')

# ubah kode HASC_1 mengikuti kode dalam CRPOI_df
brazil_states['HASC_1'] = brazil_states['HASC_1'].str[-2:]

# menambahkan date input
min_date = all_df['order_purchase_timestamp'].min()
max_date = all_df['order_purchase_timestamp'].max()

with st.sidebar:
    # Menambahkan contoh logo
    st.image("dashboard/logo_example.png")
    
    # Pilih rentang waktu dengan date input
    date_range = st.date_input(
        label='Rentang Waktu',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

    # Tambahkan Try-Except untuk menangani error jika hanya satu tanggal yang dipilih
    if len(date_range) == 2:
        start_date, end_date = date_range
    elif len(date_range) == 1:
        start_date = date_range[0]
        end_date = max_date  # Jika end_date tidak terisi, gunakan max_date
    else:
        start_date, end_date = min_date, max_date


main_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) & (all_df["order_purchase_timestamp"] <= str(end_date))]

# fungsi untuk memfilter data
def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule='D', on='order_purchase_timestamp').agg({
        "order_id": "nunique",
        "price": "sum"
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
        "order_id": "order_count",
        "price": "revenue"
    }, inplace=True)
    
    return daily_orders_df

def create_daily_items_order_df(df):
    daily_items_order_df = df.resample(rule='D', on='order_purchase_timestamp').agg({
        "order_item_id": "count",
        "price":'sum'
    })
    daily_items_order_df = daily_items_order_df.reset_index()
    daily_items_order_df.rename(columns={
        "order_item_id": "item_count",
        "price": "revenue"
    }, inplace=True)
    return daily_items_order_df

def create_bystate_df(df):
    bystate_df = df.groupby(by="customer_state").agg({
        'customer_id':'nunique',
        'order_item_id':'sum',
        'price':'sum',
        'freight_value':'sum',
        }).reset_index()
    
    merge_bystate_df = pd.merge(
        left=bystate_df,
        right=brazil_states,
        how='left',
        left_on='customer_state',
        right_on='HASC_1'
    )
    
    merge_bystate_df.rename(columns={
        "customer_id": "customer_count",
        "order_item_id":"item_count",
        'price':'revenue',
        'freight_value':'cost_delivery'
    }, inplace=True)

    return merge_bystate_df

def create_bystate_recent_df(df):
    recent_year = df['order_purchase_timestamp'].dt.year.max()
    recent_year_df = df[df['order_purchase_timestamp'].dt.year == recent_year]

    bystate_recent_df = recent_year_df.groupby(by="customer_state").agg({
        'customer_id':'nunique',
        'order_item_id':'sum',
        'price':'sum',
        'freight_value':'sum',
        }).reset_index()
    
    merged_geo_order_recent_df = pd.merge(
        left=bystate_recent_df,
        right=brazil_states,
        how='left',
        left_on='customer_state',
        right_on='HASC_1'
    )
    
    merged_geo_order_recent_df.rename(columns={
        "customer_id": "customer_count",
        "order_item_id":"item_count",
        'price':'revenue',
        'freight_value':'cost_delivery'
    }, inplace=True)

    return merged_geo_order_recent_df

def create_rship_df(df):
    rship_df = df.groupby(['shipping_time_bins', 'review_score']).agg({
        'order_item_id': 'count'
    }).reset_index()

    rship_df.rename(columns={
        "order_item_id":"item_count"
    }, inplace=True)

    labels = ['0-7 hari', '7-14 hari', '14-21 hari', '21-28 hari', '28+ hari']
    # Mengatur 'shipping_time_bins' sebagai kategori dengan urutan tertentu
    rship_df['shipping_time_bins'] = pd.Categorical(rship_df['shipping_time_bins'], 
        categories=labels, ordered=True)

    return rship_df

def create_bycategory_df(df):
    bycategory = df.groupby(by="product_category_name").agg({
        'review_score':'mean',
        'order_item_id':'count',
        'price':'sum',
    })
    bycategory.rename(columns={
        "order_item_id":"item_count",
        'price':'revenue'
    }, inplace=True)

    return bycategory

def create_bypaymenttype_df(df):
    bypaymenttype = df.groupby(by='payment_type').agg({
        'customer_id':'nunique',
        'order_item_id':'count', # kenapa count? karena setiap pesanan barang memunculkan baris baru dan dilabeli dengan angka yang terus bertambah
        'price':'sum'
    })

    total_orders = df['order_item_id'].count()
    total_transaction = df['customer_id'].nunique()
    total_price = df['price'].sum()

    bypaymenttype['percentage_price'] = (bypaymenttype['price'] / total_price) * 100
    bypaymenttype['percentage'] = (bypaymenttype['order_item_id'] / total_orders) * 100
    bypaymenttype['percentage_transaction'] = (bypaymenttype['customer_id'] / total_transaction) * 100

    bypaymenttype.rename(columns={
        'customer_id':'transaction_count',
        'order_item_id':'item_count',
        'price':'revenue'
    }, inplace=True)

    return bypaymenttype

# memanggil fungsi-fungsi tersebut
daily_orders_df = create_daily_orders_df(main_df)
daily_items_order_df = create_daily_items_order_df(main_df)
bystate_df = create_bystate_df(main_df)
bystate_recent_df = create_bystate_recent_df(main_df)
rship_df = create_rship_df(main_df)
bycategory_df = create_bycategory_df(main_df)
bypaymenttype_df = create_bypaymenttype_df(main_df)

st.header('Submission Belajar Analisis Data Python :notebook:')
st.subheader('Daily Transaction and Item Orders')
col1, col2, col3 = st.columns(3)

with col1:
    total_transaction = daily_orders_df.order_count.sum()
    st.metric("Total Transaction", value=total_transaction)

with col2:
    total_items_order = daily_items_order_df.item_count.sum()
    st.metric("Total Items Order", value=total_items_order)

with col3:
    total_revenue = format_currency(daily_orders_df.revenue.sum(), "R$",locale='pt_BR')
    st.metric("Total Revenue", value=total_revenue)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_orders_df["order_purchase_timestamp"],
    daily_orders_df["order_count"],
    marker='o', 
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)

ax.set_title('Daily Transaction')
ax.set_xlabel('Order Purchase Timestamp', fontsize=15)
ax.set_ylabel('Order Count', fontsize=15, color="#90CAF9")
ax.tick_params(axis='y', labelsize=15, colors="#90CAF9")
ax.tick_params(axis='x', labelsize=15)

# Membuat twin axis untuk item_count
ax2 = ax.twinx()
ax2.plot(
    daily_items_order_df["order_purchase_timestamp"],
    daily_items_order_df["item_count"],
    marker='x', 
    linewidth=2,
    color="#FFAB91",
    label='Item Count'
)
ax2.set_ylabel('Item Count', fontsize=15, color="#FFAB91")
ax2.tick_params(axis='y', labelsize=15, colors="#FFAB91")
 
st.pyplot(fig)

# Pertanyaan 1
st.subheader('Correlation between Review Score and Order by Product Category')

font_title = 80
font_size = 60
label_size = 40

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(45, 25))
 
colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
top5_category = bycategory_df.sort_values(by='review_score', ascending=False).head(5)
bot5_category = bycategory_df.sort_values(by='review_score', ascending=True).head(5)

top5_category_by_item = bycategory_df.sort_values(by='item_count', ascending=False).head(5)
bot5_category_by_item = bycategory_df.sort_values(by='item_count', ascending=True).head(5)

# top and bottom mean review score category berdasarkan order_item_id
mean_review_category = [top5_category_by_item['review_score'].mean(),bot5_category_by_item['review_score'].mean()]
label_category = ['Top 5 Mean Review Category by Item','Bot 5 Mean Review Category by Item']

sns.barplot(x="review_score", y="product_category_name", data= top5_category, palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Number of Sales", fontsize=label_size)
ax[0].set_title("Best Performing Product Category", loc="center", fontsize=font_title)
ax[0].tick_params(axis='y', labelsize=label_size)
ax[0].tick_params(axis='x', labelsize=label_size)
for i in range(top5_category.shape[0]):
    ax[0].text(top5_category["review_score"].iloc[i] - 0.4, i, f'{top5_category["review_score"].iloc[i]:.2f}', va='center', fontsize=label_size) 

sns.barplot(x="review_score", y="product_category_name", data=bot5_category, palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Number of Sales", fontsize=label_size)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Worst Performing Product Category", loc="center", fontsize=font_title)
ax[1].tick_params(axis='y', labelsize=label_size)
ax[1].tick_params(axis='x', labelsize=label_size)
for i in range(bot5_category.shape[0]):
    ax[1].text(bot5_category["review_score"].iloc[i], i, f'{bot5_category["review_score"].iloc[i]:.2f}', va='center', fontsize=label_size) 

st.pyplot(fig)

fig, ax = plt.subplots(figsize=(15, 15))
ax.scatter(bycategory_df['review_score'],bycategory_df['item_count'])
ax.set_xlabel('Mean Review Rating by Category (1-5)', fontsize=label_size)
ax.set_ylabel('Number of Order Items per Category', fontsize=label_size)
ax.set_xlim(left=0)
ax.set_title('Scatter Plot for Order Items vs. Mean Review Rating by Product Category',fontsize=label_size)
st.pyplot(fig)

# Rata-rata nilai ulasan 5 kategori produk teratas dan terbawah berdasarkan order_item_id 
fig, ax = plt.subplots(figsize=(15,15))
sns.barplot(x=label_category,y=mean_review_category,ax=ax)
ax.set_ylabel('Average Review Score')
ax.set_ylim(top=5)
ax.set_title('Comparison of Average Review Score: Top 5 vs Bottom 5',fontsize=font_size)
for i, value in enumerate(mean_review_category):
    ax.text(i, value + 0.1, f'{value:.2f}', ha='center', va='top', fontsize=12)
st.pyplot(fig)

# Pertanyaan 2
st.subheader('Penjualan Berdasarkan State')

gdf_plot = gpd.GeoDataFrame(bystate_df, geometry='geometry')
gdf_recent_plot = gpd.GeoDataFrame(bystate_recent_df, geometry='geometry')

#menambahkan keterangan 5 state yang memiliki item_count terbanyak
top_5_states = gdf_plot.nlargest(5, 'item_count')
top_5_recent_states = gdf_recent_plot.nlargest(5, 'item_count')

# Plot peta berdasarkan jumlah pesanan (item_count) per state
# keseluruhan data
fig, ax = plt.subplots(1,2, figsize=(12,4))

gdf_plot.plot(column='item_count', legend=True, cmap='OrRd', ax=ax[0], edgecolor='black',linewidth=0.5)
for x, y, label in zip(top_5_states.geometry.centroid.x, 
                        top_5_states.geometry.centroid.y, 
                        top_5_states['NAME_1']):
    ax[0].text(x, y, f'{label}', fontsize=5, ha='center', color='white',
            bbox=dict(facecolor='black', alpha=0.5))
    
ax[0].set_title('Distribution of Goods Orders by State in Brazil', fontsize=8)

# berdasarkan tahun terkini (2018)
gdf_recent_plot.plot(column='item_count', legend=True, cmap='OrRd', ax=ax[1], edgecolor='black',linewidth=0.5)
for x, y, label in zip(top_5_states.geometry.centroid.x, 
                              top_5_states.geometry.centroid.y, 
                              top_5_states['NAME_1']):
    ax[1].text(x, y, f'{label}', fontsize=5, ha='center', color='white',
            bbox=dict(facecolor='black', alpha=0.5))

ax[1].set_title('Distribution of Goods Orders by State in Brazil (2018)', fontsize=8)
st.pyplot(fig)

# Membuat plot untuk 5 state dengan jumlah order terbanyak
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))

# Plot negara bagian dengan jumlah order terbanyak
top_5_states.plot(column='item_count', cmap='OrRd', linewidth=0.5, ax=ax2[0], edgecolor='black', legend=True)
top_5_recent_states.plot(column='item_count', cmap='OrRd', linewidth=0.5, ax=ax2[1], edgecolor='black', legend=True)

## Menambahkan nama state dan nilai order pada plot
# seluruh data
for x, y, label, value in zip(top_5_states.geometry.centroid.x, 
                              top_5_states.geometry.centroid.y, 
                              top_5_states['NAME_1'], 
                              top_5_states['item_count']):
    ax2[0].text(x, y, f'{label}\n{value}', fontsize=5, ha='center', color='white',
             bbox=dict(facecolor='black', alpha=0.5))
ax2[0].set_title('Top 5 States with Most Order Items', fontsize=8)

# berdasarkan tahun 2018
for x, y, label, value in zip(top_5_recent_states.geometry.centroid.x, 
                              top_5_recent_states.geometry.centroid.y, 
                              top_5_recent_states['NAME_1'], 
                              top_5_recent_states['item_count']):
    ax2[1].text(x, y, f'{label}\n{value}', fontsize=5, ha='center', color='white',
             bbox=dict(facecolor='black', alpha=0.5))
ax2[1].set_title('Top 5 States with Most Order Items (2018)', fontsize=8)

st.pyplot(fig2)

# Pertanyaan 3
st.subheader('Pengaruh lama pengiriman terhadap penilaian produk')

fig, ax = plt.subplots(1,1,figsize=(12,6))
sns.barplot(x='shipping_time_bins', y='item_count', hue='review_score', data=rship_df)
ax.set_title('Kelompok Waktu Pengiriman')
ax.set_ylabel('Jumlah Order')
ax.legend(title='Review Score')
ax.grid()
st.pyplot(fig)

# Pertanyaan 4
st.subheader('Tipe pembayaran apa yang paling banyak digunakan oleh pelanggan dalam melakukan transaksi?')

col1, col2 = st.columns(2)

with col1:
    most_payment_type = bypaymenttype_df['item_count'].idxmax()
    st.metric("Most Payment Type", value=most_payment_type)

with col2:
    total_items_order = bypaymenttype_df['item_count'].sort_values(ascending=False).head(1)
    st.metric("Total Items Order", value=total_items_order)

col3, col4 = st.columns(2)
with col3:
    total_transaction = bypaymenttype_df['transaction_count'].sort_values(ascending=False).head(1)
    st.metric("Total Transaction", value=total_transaction)

with col4:
    total_revenue = format_currency(bypaymenttype_df['revenue'].sort_values(ascending=False).head(1).values[0], "R$",locale='pt_BR')
    st.metric("Total Revenue", value=total_revenue)

fig, ax = plt.subplots(2,1,figsize=(12,12))
barplot = sns.barplot(x='item_count',y='payment_type',data=bypaymenttype_df, ax=ax[0])
ax[0].set_title('Total Pesanan Barang per Jenis Pembayaran')
ax[0].set_xlabel('Jumlah Pesanan')
ax[0].set_ylabel('Jenis Pembayaran')

# Menambahkan label persentase di atas bar
for index, row in bypaymenttype_df.iterrows():
    barplot.text(row['item_count'], index, f"{row['percentage']:.2f}%", color='black', ha="left", va="center")

barplot = sns.barplot(x='transaction_count',y='payment_type',data=bypaymenttype_df, ax=ax[1])
ax[1].set_title('Total Transaksi per Jenis Pembayaran')
ax[1].set_xlabel('Jumlah Transaksi')
ax[1].set_ylabel('Jenis Pembayaran')

# Menambahkan label persentase di atas bar
for index, row in bypaymenttype_df.iterrows():
    barplot.text(row['transaction_count'], index, f"{row['percentage_transaction']:.2f}%", color='black', ha="left", va="center")

st.pyplot(fig)


# Pertanyaan 5
st.subheader('Tipe pembayaran apa yang paling banyak digunakan oleh pelanggan dalam melakukan transaksi?')

fig, ax = plt.subplots(1,1,figsize=(12,6))
barplot_order = sns.barplot(x='revenue',y='payment_type',data=bypaymenttype_df, ax=ax)
ax.set_title('Total Nilai Transaksi per Jenis Pembayaran')
ax.set_xlabel('Total Nilai Transaksi')
ax.set_ylabel('Jenis Pembayaran')

# Menambahkan label persentase di atas bar
for index, row in bypaymenttype_df.iterrows():
    barplot_order.text(row['revenue'], index, f"{row['percentage_price']:.2f}%", color='black', ha="left", va="center")

st.pyplot(fig)